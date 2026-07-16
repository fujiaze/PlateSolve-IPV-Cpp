// ============================================================================
// vm43_fit.cpp - V4.3 WcsFitter 模块 (Phase E / IRM Step 4)
//
// 职责: 分层 Huber 稳健 LSQ 拟合: CD → 径向畸变 → SIP (BIC 选阶)
// 从 V4.2 wf_core.cpp 升级:
//   - 普通 LSQ → Huber 稳健 LSQ (IRLS)
//   - δ = 1.345 × MAD(r) × 1.4826
//   - 权重 w_i = min(1, δ/|r_i|)
//   - 3-5 轮 IRLS 收敛
//
// 分层结构 (保持 V4.2):
//   Layer 0: Umeyama SVD → CD 矩阵 (标准 WCS, 无 1/cos(Dec) 因子)
//   Layer 1: 像素残差 MAD 剔除 outlier (3 轮) → 6 参数全仿射 (Huber LSQ) → 更新 CD/CRVAL
//   Layer 2: BIC 选择 SIP 阶数 (2-max_order, 高阶需 BIC 差 > 2 才选) → Huber LSQ
//
// 依赖: Eigen3 (JacobiSVD + LDLT 最小二乘)
// 约束: C++17, 中文注释, UTF-8 编码
// ============================================================================

#include "vm43_internal.h"

#include <Eigen/Dense>
#include <vector>
#include <cmath>
#include <cstring>
#include <algorithm>
#include <string>

namespace v43 {

// ============================================================================
// 常量
// ============================================================================
static constexpr double DEGTORAD = 0.017453292519943295;
static constexpr double RADTODEG = 57.295779513082323;

// ============================================================================
// 工具函数: 多项式项数
// 给定最大阶数, 返回多项式项数 = (max_order+1)(max_order+2)/2
// ============================================================================
static int poly_nterms(int max_order) {
    return (max_order + 1) * (max_order + 2) / 2;
}

// ============================================================================
// Umeyama SVD: 拟合 dst = s·R·src + t (2D 相似变换)
// 输入: src[n×2], dst[n×2] (角秒坐标)
// 输出: SimTransform (s, θ, tx, ty)
// ============================================================================
static SimTransform umeyama(const double* src, const double* dst, int n) {
    SimTransform r;
    r.valid = false; r.s = 1; r.theta = 0; r.tx = 0; r.ty = 0;
    if (n < 2) return r;

    using M2 = Eigen::Matrix2d;
    using V2 = Eigen::Vector2d;

    // 质心
    V2 ms = V2::Zero(), md = V2::Zero();
    for (int i = 0; i < n; ++i) {
        ms += V2(src[i * 2], src[i * 2 + 1]);
        md += V2(dst[i * 2], dst[i * 2 + 1]);
    }
    ms /= n; md /= n;

    // 去质心坐标
    Eigen::MatrixXd sc(2, n), dc(2, n);
    for (int i = 0; i < n; ++i) {
        sc(0, i) = src[i * 2]     - ms(0);
        sc(1, i) = src[i * 2 + 1] - ms(1);
        dc(0, i) = dst[i * 2]     - md(0);
        dc(1, i) = dst[i * 2 + 1] - md(1);
    }

    // SVD 求解旋转
    M2 H = sc * dc.transpose();
    Eigen::JacobiSVD<M2> svd(H, Eigen::ComputeFullU | Eigen::ComputeFullV);
    double det = (svd.matrixV().transpose() * svd.matrixU().transpose()).determinant();
    V2 Sv = V2::Ones(); Sv(1) = det;
    M2 R = svd.matrixV() * Sv.asDiagonal() * svd.matrixU().transpose();

    // 尺度
    double tr = sc.colwise().squaredNorm().sum();
    if (tr < 1e-15) return r;
    double s = svd.singularValues().dot(Sv) / tr;
    if (std::abs(s - 1.0) >= 0.1) return r;  // 尺度变化 > 10% 视为异常

    // 平移
    double th = std::atan2(R(1, 0), R(0, 0));
    V2 t = md - s * R * ms;
    r.s = s; r.theta = th; r.tx = t(0); r.ty = t(1); r.valid = true;
    return r;
}

// ============================================================================
// Huber 稳健 LSQ (IRLS 迭代加权最小二乘)
//
// Huber 损失:
//   ρ(r) = r²              if |r| ≤ δ
//        = 2δ|r| - δ²      if |r| > δ
//   δ = delta_factor × σ̂  (delta_factor=1.345 → 95% 渐近效率)
//   σ̂ = MAD(r) × 1.4826
//
// IRLS:
//   1. w_i = 1 (初始)
//   2. x = (A^T W A)^-1 A^T W b (加权 LSQ)
//   3. r_i = |A·x - b| (残差)
//   4. σ̂ = median(|r - median(r)|) × 1.4826
//   5. δ = delta_factor × σ̂
//   6. w_i = min(1, δ/|r_i|)
//   7. 收敛: 权重变化 < 1e-6 或 iter ≥ 5
// ============================================================================
static Eigen::VectorXd huber_lsq(const Eigen::MatrixXd& A, const Eigen::VectorXd& b,
                                  double delta_factor = 1.345) {
    int n = A.rows();
    if (n == 0) return Eigen::VectorXd();
    Eigen::VectorXd weights = Eigen::VectorXd::Ones(n);
    Eigen::VectorXd x;

    for (int iter = 0; iter < 5; ++iter) {
        // 加权 LSQ: x = (A^T W A)^-1 A^T W b
        Eigen::MatrixXd WA = A.transpose() * weights.asDiagonal();
        x = (WA * A).ldlt().solve(WA * b);

        // 计算残差 |r_i|
        Eigen::VectorXd r = (A * x - b).cwiseAbs();

        // MAD 估计 σ̂: median(|r - median(r)|) × 1.4826
        std::vector<double> r_vec(r.data(), r.data() + n);
        std::sort(r_vec.begin(), r_vec.end());
        double median_r = r_vec[n / 2];
        std::vector<double> dev(n);
        for (int i = 0; i < n; ++i) dev[i] = std::abs(r[i] - median_r);
        std::sort(dev.begin(), dev.end());
        double sigma_hat = dev[n / 2] * 1.4826;  // σ̂ = MAD × 1.4826

        // δ = delta_factor × σ̂
        double delta = delta_factor * sigma_hat;
        if (delta < 1e-10) break;  // 防止除零

        // 更新权重 w_i = min(1, δ/|r_i|)
        Eigen::VectorXd new_weights(n);
        for (int i = 0; i < n; ++i) {
            new_weights[i] = std::min(1.0, delta / std::max(r[i], 1e-10));
        }

        // 收敛判定: 权重变化 < 1e-6
        double w_change = (new_weights - weights).cwiseAbs().maxCoeff();
        weights = new_weights;
        if (w_change < 1e-6) break;
    }
    return x;
}

// ============================================================================
// BIC 计算: BIC = n·ln(RSS/n) + k·ln(n)
// RSS 下限防止完美拟合时 ln(RSS/n) → -∞
// ============================================================================
static double compute_bic(double rss, int n, int k) {
    if (n <= 0) return 1e30;
    double rss_floor = std::max(rss, 1e-10 * n);
    return n * std::log(rss_floor / n) + k * std::log((double)n);
}

// ============================================================================
// vm43_fit - 主入口: 分层 Huber 稳健 LSQ 拟合
//
// 输入: U (图像侧角秒) + W (星表侧角秒) + pairs + 中心指向 + 焦距/像元 + 参数
// 输出: WcsFitResult (CD + CRVAL + CRPIX + SIP + RMS)
// 返回: 0=成功, -1=失败
// ============================================================================
int vm43_fit(
    const std::vector<StarPoint>& U,
    const std::vector<StarPoint>& W,
    const std::vector<MatchPair>& pairs,
    double ra0, double dec0,
    double focal_length_mm,
    double pixel_size_um,
    int img_width, int img_height,
    int sip_max_order,
    const VM43SolveParams& params,
    WcsFitResult& output,
    Logger* logger)
{
    // --- 初始化 output ---
    std::memset(&output, 0, sizeof(output));
    output.success = false;

    // --- 辅助日志宏 ---
    #define LOG_INFO(msg)  do { if (logger) logger->info(msg);  } while(0)
    #define LOG_WARN(msg)  do { if (logger) logger->warn(msg);  } while(0)
    #define LOG_ERROR(msg) do { if (logger) logger->error(msg); } while(0)

    int n_pairs = (int)pairs.size();
    LOG_INFO("=== vm43_fit 开始 (n_pairs=" + std::to_string(n_pairs) +
             ", N_img=" + std::to_string((int)U.size()) +
             ", M=" + std::to_string((int)W.size()) + ") ===");

    // --- 边界检查 ---
    if (n_pairs < 2) {
        LOG_WARN("vm43_fit: n_pairs < 2, 无法拟合");
        return -1;
    }
    if (focal_length_mm <= 0 || pixel_size_um <= 0) {
        LOG_ERROR("vm43_fit: focal_length_mm 或 pixel_size_um <= 0");
        return -1;
    }

    // --- 计算像素尺度 s0 (角秒/像素) ---
    // s0 = 206264.806247 × (pixel_size_um / 1000) / focal_length_mm
    //    = 206.264806247 × pixel_size_um / focal_length_mm
    double s0 = 206.264806247 * pixel_size_um / focal_length_mm;
    if (s0 <= 0) {
        LOG_ERROR("vm43_fit: s0 <= 0");
        return -1;
    }

    // --- 提取参数 ---
    int    max_order = sip_max_order > 0 ? sip_max_order : 4;
    int    skip_sip = params.skip_sip;
    double img_w = (double)img_width;
    double img_h = (double)img_height;
    double center_ra = ra0;
    double center_dec = dec0;
    int    irm_sip_min_pairs = params.irm_sip_min_pairs > 0 ? params.irm_sip_min_pairs : 30;
    double huber_delta_factor = params.irm_huber_delta_factor > 0 ? params.irm_huber_delta_factor : 1.345;

    double cos_dec0 = std::cos(center_dec * DEGTORAD);
    if (cos_dec0 < 1e-10) cos_dec0 = 1e-10;

    int M_D = n_pairs;
    // CRPIX: 1-based (FITS 标准)
    double crpix_x = img_w / 2.0 + 1.0;
    double crpix_y = img_h / 2.0 + 1.0;

    output.crval[0] = center_ra;
    output.crval[1] = center_dec;
    output.crpix[0] = crpix_x;
    output.crpix[1] = crpix_y;

    // ============================================================
    // Layer 0: Umeyama SVD → CD 矩阵
    //   拟合 U = s·R·W + t (角秒空间)
    //   CD = [s3600·ct, -s3600·st, -s3600·st, -s3600·ct]
    //   其中 s3600 = s0 / (s × 3600)
    // ============================================================
    LOG_INFO("Layer 0: Umeyama SVD → CD 矩阵");

    std::vector<double> ws0(M_D * 2), us0(M_D * 2);
    for (int i = 0; i < M_D; ++i) {
        int ui = pairs[i].u, wi = pairs[i].w;
        ws0[i * 2]     = W[wi].x;
        ws0[i * 2 + 1] = W[wi].y;
        us0[i * 2]     = U[ui].x;
        us0[i * 2 + 1] = U[ui].y;
    }

    SimTransform au0 = umeyama(ws0.data(), us0.data(), M_D);
    if (!au0.valid) {
        LOG_ERROR("Layer 0: Umeyama 拟合失败 (s 偏离 1.0 > 10% 或点数不足)");
        return -1;
    }

    double s3600 = s0 / (au0.s * 3600.0);
    double ct = std::cos(au0.theta), st = std::sin(au0.theta);
    // 标准 WCS CD 矩阵 (无 1/cos(Dec) 因子, 无 flip):
    //   CD = [s3600·ct, -s3600·st, -s3600·st, -s3600·ct]
    output.cd.cd11 =  s3600 * ct;
    output.cd.cd12 = -s3600 * st;
    output.cd.cd21 = -s3600 * st;
    output.cd.cd22 = -s3600 * ct;

    // Layer 0 RMS (像素)
    double rms_l0 = 0;
    for (int i = 0; i < M_D; ++i) {
        int ui = pairs[i].u, wi = pairs[i].w;
        double Wx = W[wi].x, Wy = W[wi].y;
        double pred_ux = au0.s * (ct * Wx - st * Wy) + au0.tx;
        double pred_uy = au0.s * (st * Wx + ct * Wy) + au0.ty;
        double dx = (pred_ux - U[ui].x) / s0;
        double dy = (pred_uy - U[ui].y) / s0;
        rms_l0 += dx * dx + dy * dy;
    }
    rms_l0 = std::sqrt(rms_l0 / M_D);

    LOG_INFO("Layer 0: s=" + std::to_string(au0.s) +
             " theta=" + std::to_string(au0.theta * RADTODEG) +
             " RMS=" + std::to_string(rms_l0) + "px");

    // CD 逆矩阵 (用于 Layer 1 将 W 变换到像素空间)
    double cd_det = output.cd.cd11 * output.cd.cd22 - output.cd.cd12 * output.cd.cd21;
    if (std::abs(cd_det) < 1e-20) {
        LOG_ERROR("Layer 0: CD 矩阵行列式接近 0");
        return -1;
    }
    double cdi2[4] = {
        output.cd.cd22 / cd_det, -output.cd.cd12 / cd_det,
        -output.cd.cd21 / cd_det,  output.cd.cd11 / cd_det
    };

    // ============================================================
    // Layer 1: MAD 剔除 outlier (3 轮) + Huber 稳健全仿射
    //   像素空间: ssx = Ux/s0, ssy = -Uy/s0 (Y 轴翻转)
    //   CD 变换预测: ddx = cdi2·W/3600 + t_ume_px
    //   MAD 阈值: max(5px, 3×1.4826×MAD)
    //   Huber 仿射: ssx = a00·ddx + a01·ddy + tx, ssy = a10·ddx + a11·ddy + ty
    //   更新: CD' = CD·A^{-1}, CRVAL = center - CD'·t / cos(Dec)
    // ============================================================
    LOG_INFO("Layer 1: MAD + Huber 全仿射");

    std::vector<double> ssx(M_D), ssy(M_D), ddx(M_D), ddy(M_D);
    double tx_ume_px = au0.tx / s0;
    double ty_ume_px = au0.ty / s0;
    for (int i = 0; i < M_D; ++i) {
        int ui = pairs[i].u, wi = pairs[i].w;
        // 图像像素偏移 (原点在 CRPIX)
        ssx[i] =  U[ui].x / s0;    // X: 右为正
        ssy[i] = -U[ui].y / s0;    // Y: 翻转 (U_y 向上=Dec, 像素_y 向下)
        // CD 变换预测的像素偏移
        double Wx = W[wi].x, Wy = W[wi].y;
        ddx[i] = cdi2[0] * Wx / 3600.0 + cdi2[1] * Wy / 3600.0 + tx_ume_px;
        ddy[i] = cdi2[2] * Wx / 3600.0 + cdi2[3] * Wy / 3600.0 + ty_ume_px;
    }

    // MAD 迭代剔除 (3 轮)
    std::vector<bool> keep(M_D, true);
    for (int mad_iter = 0; mad_iter < 3; ++mad_iter) {
        std::vector<double> rlist;
        rlist.reserve(M_D);
        for (int i = 0; i < M_D; ++i) {
            if (!keep[i]) continue;
            double rx = ddx[i] - ssx[i], ry = ddy[i] - ssy[i];
            rlist.push_back(std::sqrt(rx * rx + ry * ry));
        }
        if (rlist.size() < 10) break;

        std::sort(rlist.begin(), rlist.end());
        double med = rlist[rlist.size() / 2];
        double mad_sigma = 1.4826 * med;  // MAD → σ
        double thresh = std::max(5.0, 3.0 * mad_sigma);

        int removed = 0;
        for (int i = 0; i < M_D; ++i) {
            if (!keep[i]) continue;
            double rx = ddx[i] - ssx[i], ry = ddy[i] - ssy[i];
            if (std::sqrt(rx * rx + ry * ry) > thresh) {
                keep[i] = false;
                removed++;
            }
        }
        int n_keep = (int)std::count(keep.begin(), keep.end(), true);
        LOG_INFO("Layer 1 MAD 轮" + std::to_string(mad_iter) +
                 ": keep=" + std::to_string(n_keep) +
                 " thresh=" + std::to_string(thresh) + "px" +
                 " removed=" + std::to_string(removed));
        if (removed == 0) break;
    }

    int M_clean = (int)std::count(keep.begin(), keep.end(), true);
    if (M_clean < 5) {
        LOG_WARN("Layer 1: MAD 后点数不足 (" + std::to_string(M_clean) + " < 5)");
        output.n_pairs = M_clean;
        output.rms_px = rms_l0;
        output.rms_arcsec = rms_l0 * s0;
        output.sip.order = 0;
        return -1;
    }

    // 构造 Huber LSQ 方程组 (2M_clean × 6)
    //   dst_x = a00·sx + a01·sy + tx
    //   dst_y = a10·sx + a11·sy + ty
    //   待求: [a00, a01, tx, a10, a11, ty]
    Eigen::MatrixXd L(M_clean * 2, 6);
    Eigen::VectorXd Rvec(M_clean * 2);
    {
        int k = 0;
        for (int i = 0; i < M_D; ++i) {
            if (!keep[i]) continue;
            L(k * 2,     0) = ddx[i]; L(k * 2,     1) = ddy[i]; L(k * 2,     2) = 1;
            L(k * 2,     3) = 0;      L(k * 2,     4) = 0;      L(k * 2,     5) = 0;
            L(k * 2 + 1, 0) = 0;      L(k * 2 + 1, 1) = 0;      L(k * 2 + 1, 2) = 0;
            L(k * 2 + 1, 3) = ddx[i]; L(k * 2 + 1, 4) = ddy[i]; L(k * 2 + 1, 5) = 1;
            Rvec(k * 2)     = ssx[i];
            Rvec(k * 2 + 1) = ssy[i];
            k++;
        }
    }

    // Huber 稳健 LSQ 求解 (V4.3 升级: V4.2 用普通 LSQ)
    Eigen::VectorXd ab = huber_lsq(L, Rvec, huber_delta_factor);
    double a00 = ab(0), a01 = ab(1), tx_aff = ab(2);
    double a10 = ab(3), a11 = ab(4), ty_aff = ab(5);

    // 更新 CD: CD' = CD · A^{-1}
    double idet = a00 * a11 - a01 * a10;
    if (std::abs(idet) < 1e-20) {
        LOG_ERROR("Layer 1: 仿射矩阵行列式接近 0");
        output.n_pairs = M_clean;
        output.rms_px = rms_l0;
        output.rms_arcsec = rms_l0 * s0;
        output.sip.order = 0;
        return -1;
    }
    double ai00 = a11 / idet, ai01 = -a01 / idet;
    double ai10 = -a10 / idet, ai11 = a00 / idet;
    double c00 = output.cd.cd11 * ai00 + output.cd.cd12 * ai10;
    double c01 = output.cd.cd11 * ai01 + output.cd.cd12 * ai11;
    double c10 = output.cd.cd21 * ai00 + output.cd.cd22 * ai10;
    double c11 = output.cd.cd21 * ai01 + output.cd.cd22 * ai11;
    output.cd.cd11 = c00; output.cd.cd12 = c01;
    output.cd.cd21 = c10; output.cd.cd22 = c11;

    // 更新 CRVAL: CRVAL = center - CD'·t / cos(Dec) (RA), center - CD'·t (Dec)
    output.crval[0] = center_ra - (c00 * tx_aff + c01 * ty_aff) / cos_dec0;
    output.crval[1] = center_dec - (c10 * tx_aff + c11 * ty_aff);

    // Layer 1 RMS (仿射后)
    double rms_l1 = 0;
    for (int i = 0; i < M_D; ++i) {
        if (!keep[i]) continue;
        double xi  = a00 * ddx[i] + a01 * ddy[i] + tx_aff;
        double eta = a10 * ddx[i] + a11 * ddy[i] + ty_aff;
        double rx = xi - ssx[i], ry = eta - ssy[i];
        rms_l1 += rx * rx + ry * ry;
        // 覆盖 ddx/ddy 为仿射预测值, 供 SIP 残差使用
        ddx[i] = xi;
        ddy[i] = eta;
    }
    rms_l1 = std::sqrt(rms_l1 / M_clean);

    LOG_INFO("Layer 1: Huber 仿射 A=[" + std::to_string(a00) + "," + std::to_string(a01) +
             ";" + std::to_string(a10) + "," + std::to_string(a11) +
             "] t=[" + std::to_string(tx_aff) + "," + std::to_string(ty_aff) +
             "]px RMS=" + std::to_string(rms_l1) + "px");

    // 设置默认结果 (无 SIP)
    output.rms_px = rms_l1;
    output.rms_arcsec = rms_l1 * s0;
    output.n_pairs = M_clean;
    output.sip.order = 0;
    output.success = true;

    // ============================================================
    // Layer 2: BIC 选择 SIP 阶数 (2-max_order) → Huber LSQ
    //   条件: skip_sip=0 且 M_clean >= irm_sip_min_pairs (默认 30)
    //   SIP 残差: ddx_affine - ssx (仿射预测 - 实际像素)
    //   归一化坐标: nu = ssx / (img_w/2), nv = ssy / (img_h/2)
    //   BIC = n·ln(RSS/n) + 2k·ln(n), k = nterms(order) - 3
    //   选阶规则: 高阶 BIC 需比低阶 BIC 低 > 2 才选 (防过拟合)
    // ============================================================
    if (skip_sip || M_clean < irm_sip_min_pairs) {
        if (skip_sip) LOG_INFO("Layer 2: 跳过 SIP (skip_sip=1)");
        else          LOG_INFO("Layer 2: 跳过 SIP (M_clean=" + std::to_string(M_clean) +
                               " < " + std::to_string(irm_sip_min_pairs) + ")");
        LOG_INFO("vm43_fit 完成: CD=[" + std::to_string(output.cd.cd11) + "," +
                 std::to_string(output.cd.cd12) + "," + std::to_string(output.cd.cd21) + "," +
                 std::to_string(output.cd.cd22) + "] sip_order=0 RMS=" +
                 std::to_string(output.rms_px) + "px");
        return 0;
    }

    LOG_INFO("Layer 2: BIC 选择 SIP 阶数 (max_order=" + std::to_string(max_order) + ")");

    // 构造 SIP 输入 (仅 clean 点)
    double xs = img_w / 2.0, ys = img_h / 2.0;
    if (xs < 1e-10) xs = 1.0;
    if (ys < 1e-10) ys = 1.0;

    std::vector<int> clean_idx;
    clean_idx.reserve(M_clean);
    std::vector<double> nu(M_clean), nv(M_clean);
    std::vector<double> rx_h(M_clean), ry_h(M_clean);
    int kc = 0;
    for (int i = 0; i < M_D; ++i) {
        if (!keep[i]) continue;
        nu[kc] = ssx[i] / xs;
        nv[kc] = ssy[i] / ys;
        rx_h[kc] = ddx[i] - ssx[i];  // 残差 x (仿射预测 - 实际)
        ry_h[kc] = ddy[i] - ssy[i];  // 残差 y
        clean_idx.push_back(i);
        kc++;
    }
    int Mc = (int)clean_idx.size();

    // BIC 基线: order 0 (仅仿射, 无 SIP 项, k=0)
    double rss0 = 0;
    for (int k = 0; k < Mc; ++k) {
        rss0 += rx_h[k] * rx_h[k] + ry_h[k] * ry_h[k];
    }
    double best_bic = compute_bic(rss0, Mc, 0);
    int    best_order = 0;
    double best_rms = rms_l1;
    double best_sA[36] = {0}, best_sB[36] = {0};

    LOG_INFO("Layer 2: BIC(order=0)=" + std::to_string(best_bic) +
             " RSS=" + std::to_string(rss0));

    // 逐阶尝试 SIP 拟合 (2 → max_order)
    for (int try_o = 2; try_o <= max_order; ++try_o) {
        int nhi = poly_nterms(try_o) - 3;  // p+q >= 2 的项数
        if (Mc <= nhi) {
            LOG_INFO("Layer 2: order=" + std::to_string(try_o) +
                     " 跳过 (Mc=" + std::to_string(Mc) + " <= nterms=" +
                     std::to_string(nhi) + ")");
            continue;
        }

        // 构造 SIP 设计矩阵 (Mc × nhi), p+q >= 2
        Eigen::MatrixXd A(Mc, nhi);
        for (int j = 0; j < Mc; ++j) {
            int co = 0;
            for (int o = 2; o <= try_o; ++o)
                for (int p = 0; p <= o; ++p)
                    A(j, co++) = std::pow(nu[j], p) * std::pow(nv[j], o - p);
        }
        Eigen::VectorXd bx(Mc), by(Mc);
        for (int j = 0; j < Mc; ++j) {
            bx(j) = rx_h[j];
            by(j) = ry_h[j];
        }

        // Huber 稳健 LSQ 拟合 SIP 残差 (V4.3 升级: V4.2 用普通 LSQ)
        Eigen::VectorXd bxa = huber_lsq(A, bx, huber_delta_factor);
        Eigen::VectorXd bya = huber_lsq(A, by, huber_delta_factor);
        if (bxa.size() == 0 || bya.size() == 0) continue;

        // 计算 RSS
        double ssq = 0;
        for (int j = 0; j < Mc; ++j) {
            double pred_x = 0, pred_y = 0;
            int co = 0;
            for (int o = 2; o <= try_o; ++o)
                for (int p = 0; p <= o; ++p) {
                    double term = std::pow(nu[j], p) * std::pow(nv[j], o - p);
                    pred_x += bxa(co) * term;
                    pred_y += bya(co) * term;
                    co++;
                }
            double ex = rx_h[j] - pred_x;
            double ey = ry_h[j] - pred_y;
            ssq += ex * ex + ey * ey;
        }
        double rms_h = std::sqrt(ssq / Mc);
        int k_params = nhi * 2;  // A 和 B 各 nhi 项
        double bic = compute_bic(ssq, Mc, k_params);

        LOG_INFO("Layer 2: order=" + std::to_string(try_o) +
                 " nterms=" + std::to_string(nhi) +
                 " RSS=" + std::to_string(ssq) +
                 " RMS=" + std::to_string(rms_h) + "px" +
                 " BIC=" + std::to_string(bic) +
                 " (ΔBIC=" + std::to_string(bic - best_bic) + ")");

        // 选阶: BIC 需比当前最优低 > 2 才选高阶 (防过拟合)
        if (bic < best_bic - 2.0) {
            best_bic = bic;
            best_order = try_o;
            best_rms = rms_h;

            // 提取并反归一化 SIP 系数
            std::memset(best_sA, 0, sizeof(best_sA));
            std::memset(best_sB, 0, sizeof(best_sB));
            for (int o = 2; o <= try_o; ++o) {
                for (int p = 0; p <= o; ++p) {
                    int q = o - p;
                    if (p >= 6 || q >= 6) continue;
                    // 查找 (p,q) 在系数向量中的索引
                    int hi = -1, cn = 0;
                    for (int oo = 2; oo <= try_o; ++oo)
                        for (int pp = 0; pp <= oo; ++pp) {
                            if (pp == p && (oo - pp) == q) { hi = cn; break; }
                            cn++;
                        }
                    if (hi < 0 || hi >= nhi) continue;
                    double nf = std::pow(xs, p) * std::pow(ys, q);
                    best_sA[p * 6 + q] = bxa(hi) / nf;
                    best_sB[p * 6 + q] = bya(hi) / nf;
                }
            }
        }
    }

    // 写入 SIP 结果
    output.sip.order = best_order;
    output.rms_px = best_rms;
    output.rms_arcsec = best_rms * s0;
    for (int i = 0; i < 36; ++i) {
        output.sip.A[i] = best_sA[i];
        output.sip.B[i] = best_sB[i];
    }

    LOG_INFO("Layer 2: 选定 sip_order=" + std::to_string(best_order) +
             " RMS=" + std::to_string(best_rms) + "px" +
             " (Layer1 RMS=" + std::to_string(rms_l1) + "px)");

    LOG_INFO("vm43_fit 完成: CD=[" + std::to_string(output.cd.cd11) + "," +
             std::to_string(output.cd.cd12) + "," + std::to_string(output.cd.cd21) + "," +
             std::to_string(output.cd.cd22) + "] CRVAL=[" +
             std::to_string(output.crval[0]) + "," + std::to_string(output.crval[1]) +
             "] sip_order=" + std::to_string(output.sip.order) +
             " RMS=" + std::to_string(output.rms_px) + "px");

    #undef LOG_INFO
    #undef LOG_WARN
    #undef LOG_ERROR

    return 0;
}

} // namespace v43
