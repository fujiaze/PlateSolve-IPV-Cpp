// ============================================================================
// wf_core.cpp - V4.2 WcsFitter 核心实现（Task 6）
//
// Phase E 分层 SIP 拟合:
//   Layer 0: Umeyama SVD → CD 矩阵 (标准 WCS, 无 1/cos(Dec) 因子)
//   Layer 1: 像素残差 MAD 剔除 outlier (3 轮) → 6 参数全仿射 → 更新 CD/CRVAL
//   Layer 2: BIC 选择 SIP 阶数 (2-4 阶, 高阶需 BIC 差 > 2 才选)
//
// 从 V4.1 fit_affine_sip_adaptive 迁移, 简化:
//   - 无 flip_mode (V4.2 VectorMatcher 已统一方向)
//   - 无 pre-filter (PairVerifier 已做 MAD 清洗)
//   - 无 JSON 输出 (WcsResult 值类型返回)
//   - BIC 增加 order-0 基线比较 + > 2 阈值防过拟合
//
// 依赖: Eigen3 (JacobiSVD + LDLT 最小二乘)
// 约束: C++17, 单线程; 中文注释, UTF-8 编码
// ============================================================================

#include "wf_api.h"
#include "v42_log.h"

#include <Eigen/Dense>
#include <vector>
#include <cmath>
#include <cstring>
#include <algorithm>
#include <string>

// ============================================================================
// 常量
// ============================================================================
static constexpr double DEGTORAD = 0.017453292519943295;
static constexpr double RADTODEG = 57.295779513082323;

// ============================================================================
// 工具函数: 多项式基索引
// ============================================================================

// poly_index: 返回 (p,q) 项在多项式基中的索引
// (V4.1 中定义但实际 SIP 拟合使用内联索引查找, 此处保留以备外部调用)
__attribute__((unused))
static int poly_index(int p, int q, int max_order) {
    int idx = 0;
    for (int o = 0; o <= max_order; ++o)
        for (int pp = 0; pp <= o; ++pp) {
            int qq = o - pp;
            if (pp == p && qq == q) return idx;
            idx++;
        }
    return -1;
}

// 给定最大阶数, 返回多项式项数 = (max_order+1)(max_order+2)/2
static int poly_nterms(int max_order) {
    return (max_order + 1) * (max_order + 2) / 2;
}

// ============================================================================
// 工具函数: 中位数 + MAD (中位绝对偏差)
// ============================================================================

// (median_mad 工具函数已内联到 Layer 1 的 MAD 迭代中, 不再单独定义)

// ============================================================================
// Umeyama SVD: 拟合 dst = s·R·src + t (2D 相似变换)
// ============================================================================

static v42::SimTransform umeyama(const double* src, const double* dst, int n) {
    v42::SimTransform r;
    r.valid = false; r.s = 1; r.theta = 0; r.tx = 0; r.ty = 0;
    if (n < 2) return r;

    using M2 = Eigen::Matrix2d;
    using V2 = Eigen::Vector2d;

    V2 ms = V2::Zero(), md = V2::Zero();
    for (int i = 0; i < n; ++i) {
        ms += V2(src[i * 2], src[i * 2 + 1]);
        md += V2(dst[i * 2], dst[i * 2 + 1]);
    }
    ms /= n; md /= n;

    Eigen::MatrixXd sc(2, n), dc(2, n);
    for (int i = 0; i < n; ++i) {
        sc(0, i) = src[i * 2]     - ms(0);
        sc(1, i) = src[i * 2 + 1] - ms(1);
        dc(0, i) = dst[i * 2]     - md(0);
        dc(1, i) = dst[i * 2 + 1] - md(1);
    }

    M2 H = sc * dc.transpose();
    Eigen::JacobiSVD<M2> svd(H, Eigen::ComputeFullU | Eigen::ComputeFullV);
    double det = (svd.matrixV().transpose() * svd.matrixU().transpose()).determinant();
    V2 Sv = V2::Ones(); Sv(1) = det;
    M2 R = svd.matrixV() * Sv.asDiagonal() * svd.matrixU().transpose();

    double tr = sc.colwise().squaredNorm().sum();
    if (tr < 1e-15) return r;
    double s = svd.singularValues().dot(Sv) / tr;
    if (std::abs(s - 1.0) >= 0.1) return r;  // 尺度变化 > 10% 视为异常

    double th = std::atan2(R(1, 0), R(0, 0));
    V2 t = md - s * R * ms;
    r.s = s; r.theta = th; r.tx = t(0); r.ty = t(1); r.valid = true;
    return r;
}

// ============================================================================
// 6 参数全仿射最小二乘: 拟合 dst = A·src + t
//   dst_x = a00·sx + a01·sy + tx
//   dst_y = a10·sx + a11·sy + ty
// 返回 [a00, a01, tx, a10, a11, ty]
// ============================================================================

static Eigen::VectorXd fit_affine(const std::vector<double>& sx,
                                   const std::vector<double>& sy,
                                   const std::vector<double>& dx,
                                   const std::vector<double>& dy) {
    int n = (int)sx.size();
    Eigen::MatrixXd L(n * 2, 6);
    Eigen::VectorXd R(n * 2);
    for (int i = 0; i < n; ++i) {
        L(i * 2,     0) = sx[i]; L(i * 2,     1) = sy[i]; L(i * 2,     2) = 1;
        L(i * 2,     3) = 0;     L(i * 2,     4) = 0;     L(i * 2,     5) = 0;
        L(i * 2 + 1, 0) = 0;     L(i * 2 + 1, 1) = 0;     L(i * 2 + 1, 2) = 0;
        L(i * 2 + 1, 3) = sx[i]; L(i * 2 + 1, 4) = sy[i]; L(i * 2 + 1, 5) = 1;
        R(i * 2)     = dx[i];
        R(i * 2 + 1) = dy[i];
    }
    return (L.transpose() * L).ldlt().solve(L.transpose() * R);
}

// ============================================================================
// SIP 残差多项式拟合: 拟合 residual = Σ coeff[p,q] · nu^p · nv^q (p+q >= 2)
// 返回系数向量 (长度 = nterms(order) - 3)
// ============================================================================

static Eigen::VectorXd fit_sip_residual(const std::vector<double>& nu,
                                         const std::vector<double>& nv,
                                         const std::vector<double>& residual,
                                         int order) {
    int n = (int)nu.size();
    int nhi = poly_nterms(order) - 3;  // p+q >= 2 的项数
    if (n <= nhi) return Eigen::VectorXd();

    Eigen::MatrixXd A(n, nhi);
    Eigen::VectorXd b(n);
    for (int j = 0; j < n; ++j) {
        b(j) = residual[j];
        int co = 0;
        for (int o = 2; o <= order; ++o)
            for (int p = 0; p <= o; ++p)
                A(j, co++) = std::pow(nu[j], p) * std::pow(nv[j], o - p);
    }
    return (A.transpose() * A).ldlt().solve(A.transpose() * b);
}

// ============================================================================
// BIC 计算: BIC = n·ln(RSS/n) + k·ln(n)
// ============================================================================

static double compute_bic(double rss, int n, int k) {
    if (n <= 0) return 1e30;
    // RSS 下限: 防止完美拟合时 ln(RSS/n) → -∞ 导致 BIC 退化为噪声主导
    // 当 RSS 极小时, BIC 应由惩罚项 k·ln(n) 主导 (阶数越高惩罚越大)
    double rss_floor = std::max(rss, 1e-10 * n);
    return n * std::log(rss_floor / n) + k * std::log((double)n);
}

// ============================================================================
// wf_fit - 主入口 (extern "C" 在 wf_api.h 中声明)
// ============================================================================

WF_API int wf_fit(
    const double* U, int N_img,
    const double* W, int M,
    const int* pairs_u, const int* pairs_w, int n_pairs,
    const WcsFitterParams* params,
    WcsResult* result)
{
    // --- 初始化 result ---
    std::memset(result, 0, sizeof(WcsResult));
    result->success = false;

    // --- 初始化日志 ---
    v42::Logger logger;
    if (params && params->log_file_path) {
        logger.init(params->log_file_path);
    }
    logger.info("=== WcsFitter 开始 (n_pairs=" + std::to_string(n_pairs) +
                ", N_img=" + std::to_string(N_img) +
                ", M=" + std::to_string(M) + ") ===");

    // --- 边界检查 ---
    if (!U || !W || !pairs_u || !pairs_w || !params || !result) {
        logger.error("wf_fit: 空指针参数");
        return 0;
    }
    if (n_pairs < 2) {
        logger.warn("wf_fit: n_pairs < 2, 无法拟合");
        return 0;
    }

    // --- 提取参数 ---
    double s0 = params->s0;
    int    sip_max_order = params->sip_max_order > 0 ? params->sip_max_order : 4;
    int    skip_sip = params->skip_sip;
    double img_w = params->img_width;
    double img_h = params->img_height;
    double center_ra = params->center_ra;
    double center_dec = params->center_dec;

    if (s0 <= 0) {
        logger.error("wf_fit: s0 <= 0");
        return 0;
    }

    double cos_dec0 = std::cos(center_dec * DEGTORAD);
    if (cos_dec0 < 1e-10) cos_dec0 = 1e-10;

    int M_D = n_pairs;
    double crpix_x = img_w / 2.0;
    double crpix_y = img_h / 2.0;

    result->crval[0] = center_ra;
    result->crval[1] = center_dec;
    result->crpix[0] = crpix_x;
    result->crpix[1] = crpix_y;

    // ============================================================
    // Layer 0: Umeyama SVD → CD 矩阵
    //   拟合 U = s·R·W + t (角秒空间)
    //   CD = [s3600·ct, -s3600·st, -s3600·st, -s3600·ct]
    //   其中 s3600 = s0 / (s × 3600)
    // ============================================================
    logger.info("Layer 0: Umeyama SVD → CD 矩阵");

    std::vector<double> ws0(M_D * 2), us0(M_D * 2);
    for (int i = 0; i < M_D; ++i) {
        int ui = pairs_u[i], wi = pairs_w[i];
        ws0[i * 2]     = W[wi * 2];
        ws0[i * 2 + 1] = W[wi * 2 + 1];
        us0[i * 2]     = U[ui * 2];
        us0[i * 2 + 1] = U[ui * 2 + 1];
    }

    v42::SimTransform au0 = umeyama(ws0.data(), us0.data(), M_D);
    if (!au0.valid) {
        logger.error("Layer 0: Umeyama 拟合失败 (s 偏离 1.0 > 10% 或点数不足)");
        return 0;
    }

    double s3600 = s0 / (au0.s * 3600.0);
    double ct = std::cos(au0.theta), st = std::sin(au0.theta);
    // 标准 WCS CD 矩阵 (无 1/cos(Dec) 因子, 无 flip):
    //   CD = [s3600·ct, -s3600·st, -s3600·st, -s3600·ct]
    result->cd[0] =  s3600 * ct;
    result->cd[1] = -s3600 * st;
    result->cd[2] = -s3600 * st;
    result->cd[3] = -s3600 * ct;

    // Layer 0 RMS (像素)
    double rms_l0 = 0;
    for (int i = 0; i < M_D; ++i) {
        int ui = pairs_u[i], wi = pairs_w[i];
        double Wx = W[wi * 2], Wy = W[wi * 2 + 1];
        double pred_ux = au0.s * (ct * Wx - st * Wy) + au0.tx;
        double pred_uy = au0.s * (st * Wx + ct * Wy) + au0.ty;
        double dx = (pred_ux - U[ui * 2])     / s0;
        double dy = (pred_uy - U[ui * 2 + 1]) / s0;
        rms_l0 += dx * dx + dy * dy;
    }
    rms_l0 = std::sqrt(rms_l0 / M_D);

    logger.info("Layer 0: s=" + std::to_string(au0.s) +
                " theta=" + std::to_string(au0.theta * RADTODEG) +
                " RMS=" + std::to_string(rms_l0) + "px");

    // CD 逆矩阵 (用于 Layer 1 将 W 变换到像素空间)
    double cd21 = result->cd[0] * result->cd[3] - result->cd[1] * result->cd[2];
    if (std::abs(cd21) < 1e-20) {
        logger.error("Layer 0: CD 矩阵行列式接近 0");
        return 0;
    }
    double cdi2[4] = {
        result->cd[3] / cd21, -result->cd[1] / cd21,
        -result->cd[2] / cd21,  result->cd[0] / cd21
    };

    // ============================================================
    // Layer 1: MAD 剔除 outlier (3 轮) + 6 参数全仿射
    //   像素空间: ssx = Ux/s0, ssy = -Uy/s0 (Y 轴翻转: U 向上, 像素向下)
    //   CD 变换预测: ddx = cdi2·W/3600 + t_ume_px
    //   MAD 阈值: max(5px, 3×1.4826×MAD)
    //   仿射: ssx = a00·ddx + a01·ddy + tx, ssy = a10·ddx + a11·ddy + ty
    //   更新: CD' = CD·A^{-1}, CRVAL = center - CD'·t / cos(Dec)
    // ============================================================
    logger.info("Layer 1: MAD + 全仿射");

    std::vector<double> ssx(M_D), ssy(M_D), ddx(M_D), ddy(M_D);
    double tx_ume_px = au0.tx / s0;
    double ty_ume_px = au0.ty / s0;
    for (int i = 0; i < M_D; ++i) {
        int ui = pairs_u[i], wi = pairs_w[i];
        // 图像像素偏移 (原点在 CRPIX)
        ssx[i] =  U[ui * 2]     / s0;   // X: 右为正
        ssy[i] = -U[ui * 2 + 1] / s0;   // Y: 翻转 (U_y 向上=Dec, 像素_y 向下)
        // CD 变换预测的像素偏移
        double Wx = W[wi * 2], Wy = W[wi * 2 + 1];
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
        logger.info("Layer 1 MAD 轮" + std::to_string(mad_iter) +
                    ": keep=" + std::to_string(n_keep) +
                    " thresh=" + std::to_string(thresh) + "px" +
                    " removed=" + std::to_string(removed));
        if (removed == 0) break;
    }

    int M_clean = (int)std::count(keep.begin(), keep.end(), true);
    if (M_clean < 5) {
        logger.warn("Layer 1: MAD 后点数不足 (" + std::to_string(M_clean) + " < 5)");
        result->n_pairs = M_clean;
        result->rms_px = rms_l0;
        result->sip_order = 0;
        result->success = false;
        return 0;
    }

    // 全仿射拟合 (仅 clean 点)
    std::vector<double> sx_c, sy_c, dx_c, dy_c;
    sx_c.reserve(M_clean); sy_c.reserve(M_clean);
    dx_c.reserve(M_clean); dy_c.reserve(M_clean);
    for (int i = 0; i < M_D; ++i) {
        if (!keep[i]) continue;
        sx_c.push_back(ddx[i]);  // 预测像素 x
        sy_c.push_back(ddy[i]);  // 预测像素 y
        dx_c.push_back(ssx[i]);  // 实际像素 x
        dy_c.push_back(ssy[i]);  // 实际像素 y
    }
    Eigen::VectorXd ab = fit_affine(sx_c, sy_c, dx_c, dy_c);
    double a00 = ab(0), a01 = ab(1), tx_aff = ab(2);
    double a10 = ab(3), a11 = ab(4), ty_aff = ab(5);

    // 更新 CD: CD' = CD · A^{-1}
    double idet = a00 * a11 - a01 * a10;
    if (std::abs(idet) < 1e-20) {
        logger.error("Layer 1: 仿射矩阵行列式接近 0");
        result->n_pairs = M_clean;
        result->rms_px = rms_l0;
        result->sip_order = 0;
        result->success = false;
        return 0;
    }
    double ai00 = a11 / idet, ai01 = -a01 / idet;
    double ai10 = -a10 / idet, ai11 = a00 / idet;
    double c00 = result->cd[0] * ai00 + result->cd[1] * ai10;
    double c01 = result->cd[0] * ai01 + result->cd[1] * ai11;
    double c10 = result->cd[2] * ai00 + result->cd[3] * ai10;
    double c11 = result->cd[2] * ai01 + result->cd[3] * ai11;
    result->cd[0] = c00; result->cd[1] = c01;
    result->cd[2] = c10; result->cd[3] = c11;

    // 更新 CRVAL: CRVAL = center - CD'·t / cos(Dec) (RA), center - CD'·t (Dec)
    result->crval[0] = center_ra - (c00 * tx_aff + c01 * ty_aff) / cos_dec0;
    result->crval[1] = center_dec - (c10 * tx_aff + c11 * ty_aff);

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

    logger.info("Layer 1: 仿射 A=[" + std::to_string(a00) + "," + std::to_string(a01) +
                ";" + std::to_string(a10) + "," + std::to_string(a11) +
                "] t=[" + std::to_string(tx_aff) + "," + std::to_string(ty_aff) +
                "]px RMS=" + std::to_string(rms_l1) + "px");

    // 设置默认结果 (无 SIP)
    result->rms_px = rms_l1;
    result->n_pairs = M_clean;
    result->sip_order = 0;
    result->success = true;

    // ============================================================
    // Layer 2: BIC 选择 SIP 阶数 (2-4 阶)
    //   条件: skip_sip=0 且 M_clean >= 20
    //   SIP 残差: ddx_affine - ssx (仿射预测 - 实际像素)
    //   归一化坐标: nu = ssx / (img_w/2), nv = ssy / (img_h/2)
    //   BIC = n·ln(RSS/n) + 2k·ln(n), k = nterms(order) - 3
    //   选阶规则: 高阶 BIC 需比低阶 BIC 低 > 2 才选高阶
    // ============================================================
    if (skip_sip || M_clean < 20) {
        if (skip_sip) logger.info("Layer 2: 跳过 SIP (skip_sip=1)");
        else          logger.info("Layer 2: 跳过 SIP (M_clean=" + std::to_string(M_clean) + " < 20)");
        logger.info("WcsFitter 完成: CD=[" + std::to_string(result->cd[0]) + "," +
                    std::to_string(result->cd[1]) + "," + std::to_string(result->cd[2]) + "," +
                    std::to_string(result->cd[3]) + "] sip_order=0 RMS=" +
                    std::to_string(result->rms_px) + "px");
        return 1;
    }

    logger.info("Layer 2: BIC 选择 SIP 阶数 (max_order=" + std::to_string(sip_max_order) + ")");

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

    logger.info("Layer 2: BIC(order=0)=" + std::to_string(best_bic) +
                " RSS=" + std::to_string(rss0));

    // 逐阶尝试 SIP 拟合 (2 → max_order)
    for (int try_o = 2; try_o <= sip_max_order; ++try_o) {
        int nhi = poly_nterms(try_o) - 3;  // p+q >= 2 的项数
        if (Mc <= nhi) {
            logger.info("Layer 2: order=" + std::to_string(try_o) +
                        " 跳过 (Mc=" + std::to_string(Mc) + " <= nterms=" +
                        std::to_string(nhi) + ")");
            continue;
        }

        // 拟合 SIP 残差 (x 和 y 分别拟合)
        Eigen::VectorXd bxa = fit_sip_residual(nu, nv, rx_h, try_o);
        Eigen::VectorXd bya = fit_sip_residual(nu, nv, ry_h, try_o);
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

        logger.info("Layer 2: order=" + std::to_string(try_o) +
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
    result->sip_order = best_order;
    result->rms_px = best_rms;
    for (int i = 0; i < 36; ++i) {
        result->sip_A[i] = best_sA[i];
        result->sip_B[i] = best_sB[i];
    }

    logger.info("Layer 2: 选定 sip_order=" + std::to_string(best_order) +
                " RMS=" + std::to_string(best_rms) + "px" +
                " (Layer1 RMS=" + std::to_string(rms_l1) + "px)");

    // 打印 SIP 系数 (前 10 个非零)
    int ndb = std::min(10, Mc);
    int np = 0;
    for (int p = 0; p < 6 && np < ndb; ++p)
        for (int q = 0; q < 6 && np < ndb; ++q) {
            if (p + q < 2 || p + q > best_order) continue;
            if (std::abs(result->sip_A[p * 6 + q]) < 1e-30 &&
                std::abs(result->sip_B[p * 6 + q]) < 1e-30) continue;
            logger.info("  SIP A[" + std::to_string(p) + "][" + std::to_string(q) +
                        "]=" + std::to_string(result->sip_A[p * 6 + q]) +
                        " B[" + std::to_string(p) + "][" + std::to_string(q) +
                        "]=" + std::to_string(result->sip_B[p * 6 + q]));
            np++;
        }

    logger.info("WcsFitter 完成: CD=[" + std::to_string(result->cd[0]) + "," +
                std::to_string(result->cd[1]) + "," + std::to_string(result->cd[2]) + "," +
                std::to_string(result->cd[3]) + "] CRVAL=[" +
                std::to_string(result->crval[0]) + "," + std::to_string(result->crval[1]) +
                "] sip_order=" + std::to_string(result->sip_order) +
                " RMS=" + std::to_string(result->rms_px) + "px");

    return 1;
}
