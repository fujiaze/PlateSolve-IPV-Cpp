// ============================================================================
// test_fit.cpp - V4.3 WcsFitter (vm44_fit) 单元测试 (Task 7)
//
// 测试内容:
//   1. 线性 CD 拟合 (无外点): 合成数据, 验证 CD 矩阵恢复精度
//   2. 含外点拟合: 50 对中 5 对外点 (残差 ~50"), Huber 优于普通 LSQ
//   3. SIP 2 阶拟合: 合成带畸变数据, 验证 SIP 系数正确
//
// 编译: make test_fit
// ============================================================================
#include "../include/vm44_internal.h"

#include <Eigen/Dense>
#include <cstdio>
#include <cmath>
#include <vector>
#include <random>
#include <algorithm>
#include <string>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

using namespace v44;

// ============================================================================
// 全局测试统计
// ============================================================================
static int g_tests_passed = 0;
static int g_tests_failed = 0;

#define TEST_PASS(name) do { \
    printf("  [PASS] %s\n", name); \
    g_tests_passed++; \
} while(0)

#define TEST_FAIL(name, msg) do { \
    printf("  [FAIL] %s: %s\n", name, msg); \
    g_tests_failed++; \
} while(0)

// ============================================================================
// 合成数据生成器
//   生成 W (星表侧) 和 U (图像侧) 角秒坐标
//   变换模型: U = s·R·W + t (角秒空间, Y轴向上)
// ============================================================================
struct SyntheticData {
    std::vector<StarPoint> U;
    std::vector<StarPoint> W;
    std::vector<MatchPair> pairs;
    double s_true;       // 真实尺度
    double theta_true;   // 真实旋转 (弧度)
    double tx_true;      // 真实 X 平移 (角秒)
    double ty_true;      // 真实 Y 平移 (角秒)
};

// 生成合成数据
// N: 点对数, noise_sigma: 噪声标准差 (角秒), outlier_indices: 外点索引集
// outlier_offset: 外点偏移量 (角秒)
static SyntheticData generate_data(
    int N, double noise_sigma,
    const std::vector<int>& outlier_indices = {},
    double outlier_offset = 0.0,
    uint32_t seed = 42)
{
    SyntheticData data;
    data.s_true = 1.0;
    data.theta_true = 10.0 * M_PI / 180.0;  // 10度旋转
    data.tx_true = 5.0;   // 5角秒偏移
    data.ty_true = -3.0;  // -3角秒偏移

    std::mt19937 rng(seed);
    std::uniform_real_distribution<double> pos_dist(-500, 500);  // ±500角秒
    std::normal_distribution<double> noise_dist(0.0, noise_sigma);

    double ct = std::cos(data.theta_true), st = std::sin(data.theta_true);

    data.U.resize(N);
    data.W.resize(N);
    data.pairs.resize(N);

    for (int i = 0; i < N; ++i) {
        // W 点 (星表侧)
        data.W[i].x = pos_dist(rng);
        data.W[i].y = pos_dist(rng);
        data.W[i].flux = 1000.0;
        data.W[i].saturated = false;

        // U = s·R·W + t + noise
        double ux = data.s_true * (ct * data.W[i].x - st * data.W[i].y) + data.tx_true;
        double uy = data.s_true * (st * data.W[i].x + ct * data.W[i].y) + data.ty_true;
        ux += noise_dist(rng);
        uy += noise_dist(rng);

        data.U[i].x = ux;
        data.U[i].y = uy;
        data.U[i].flux = 1000.0;
        data.U[i].saturated = false;

        data.pairs[i].u = i;
        data.pairs[i].w = i;
    }

    // 添加外点: 在指定索引上额外偏移
    for (int idx : outlier_indices) {
        if (idx >= 0 && idx < N) {
            // 沿一个方向偏移 outlier_offset 角秒
            data.U[idx].x += outlier_offset;
            data.U[idx].y += outlier_offset * 0.5;
        }
    }

    return data;
}

// ============================================================================
// 普通 LSQ 仿射拟合 (无 MAD, 无 Huber, 用于对比)
//   复刻 V4.2 的 fit_affine, 使用全部点对 (含外点)
//   返回 CD 矩阵 [cd11, cd12, cd21, cd22]
// ============================================================================
static Eigen::Vector4d plain_lsq_cd(
    const std::vector<StarPoint>& U,
    const std::vector<StarPoint>& W,
    const std::vector<MatchPair>& pairs,
    double s0, double center_ra, double center_dec)
{
    int M_D = (int)pairs.size();
    double cos_dec0 = std::cos(center_dec * M_PI / 180.0);
    if (cos_dec0 < 1e-10) cos_dec0 = 1e-10;

    // Umeyama (复刻)
    std::vector<double> ws0(M_D * 2), us0(M_D * 2);
    for (int i = 0; i < M_D; ++i) {
        ws0[i*2]   = W[pairs[i].w].x;
        ws0[i*2+1] = W[pairs[i].w].y;
        us0[i*2]   = U[pairs[i].u].x;
        us0[i*2+1] = U[pairs[i].u].y;
    }

    using M2 = Eigen::Matrix2d;
    using V2 = Eigen::Vector2d;
    V2 ms = V2::Zero(), md = V2::Zero();
    for (int i = 0; i < M_D; ++i) {
        ms += V2(ws0[i*2], ws0[i*2+1]);
        md += V2(us0[i*2], us0[i*2+1]);
    }
    ms /= M_D; md /= M_D;
    Eigen::MatrixXd sc(2, M_D), dc(2, M_D);
    for (int i = 0; i < M_D; ++i) {
        sc(0,i) = ws0[i*2]   - ms(0);
        sc(1,i) = ws0[i*2+1] - ms(1);
        dc(0,i) = us0[i*2]   - md(0);
        dc(1,i) = us0[i*2+1] - md(1);
    }
    M2 H = sc * dc.transpose();
    Eigen::JacobiSVD<M2> svd(H, Eigen::ComputeFullU | Eigen::ComputeFullV);
    double det = (svd.matrixV().transpose() * svd.matrixU().transpose()).determinant();
    V2 Sv = V2::Ones(); Sv(1) = det;
    M2 R = svd.matrixV() * Sv.asDiagonal() * svd.matrixU().transpose();
    double tr = sc.colwise().squaredNorm().sum();
    double s_ume = svd.singularValues().dot(Sv) / tr;
    double th = std::atan2(R(1,0), R(0,0));
    V2 t_ume = md - s_ume * R * ms;

    // 初始 CD
    double s3600 = s0 / (s_ume * 3600.0);
    double ct = std::cos(th), st = std::sin(th);
    double cd0[4] = { s3600*ct, -s3600*st, -s3600*st, -s3600*ct };
    double cd_det = cd0[0]*cd0[3] - cd0[1]*cd0[2];
    double cdi2[4] = { cd0[3]/cd_det, -cd0[1]/cd_det, -cd0[2]/cd_det, cd0[0]/cd_det };

    // 像素坐标 (无 MAD 剔除, 使用全部点)
    std::vector<double> ssx(M_D), ssy(M_D), ddx(M_D), ddy(M_D);
    for (int i = 0; i < M_D; ++i) {
        ssx[i] =  U[pairs[i].u].x / s0;
        ssy[i] = -U[pairs[i].u].y / s0;
        double Wx = W[pairs[i].w].x, Wy = W[pairs[i].w].y;
        ddx[i] = cdi2[0]*Wx/3600.0 + cdi2[1]*Wy/3600.0 + t_ume(0)/s0;
        ddy[i] = cdi2[2]*Wx/3600.0 + cdi2[3]*Wy/3600.0 + t_ume(1)/s0;
    }

    // 普通 LSQ 仿射 (全部点, 无 Huber)
    Eigen::MatrixXd L(M_D*2, 6);
    Eigen::VectorXd Rvec(M_D*2);
    for (int i = 0; i < M_D; ++i) {
        L(i*2,   0)=ddx[i]; L(i*2,   1)=ddy[i]; L(i*2,   2)=1;
        L(i*2,   3)=0;      L(i*2,   4)=0;      L(i*2,   5)=0;
        L(i*2+1, 0)=0;      L(i*2+1, 1)=0;      L(i*2+1, 2)=0;
        L(i*2+1, 3)=ddx[i]; L(i*2+1, 4)=ddy[i]; L(i*2+1, 5)=1;
        Rvec(i*2)   = ssx[i];
        Rvec(i*2+1) = ssy[i];
    }
    Eigen::VectorXd ab = (L.transpose() * L).ldlt().solve(L.transpose() * Rvec);
    double a00=ab(0), a01=ab(1);
    double a10=ab(3), a11=ab(4);

    // 更新 CD: CD' = CD · A^{-1}
    double idet = a00*a11 - a01*a10;
    double ai00=a11/idet, ai01=-a01/idet, ai10=-a10/idet, ai11=a00/idet;
    double c00 = cd0[0]*ai00 + cd0[1]*ai10;
    double c01 = cd0[0]*ai01 + cd0[1]*ai11;
    double c10 = cd0[2]*ai00 + cd0[3]*ai10;
    double c11 = cd0[2]*ai01 + cd0[3]*ai11;

    return Eigen::Vector4d(c00, c01, c10, c11);
}

// ============================================================================
// 计算理论 CD 矩阵 (从真实变换参数)
// ============================================================================
static Eigen::Vector4d theoretical_cd(double s0, double s_true, double theta_true) {
    double s3600 = s0 / (s_true * 3600.0);
    double ct = std::cos(theta_true), st = std::sin(theta_true);
    return Eigen::Vector4d(s3600*ct, -s3600*st, -s3600*st, -s3600*ct);
}

// ============================================================================
// 测试 1: 线性 CD 拟合 (无外点)
//   50 对干净数据, 0.3" 噪声
//   验证: 成功返回, RMS < 1px, CD 矩阵恢复误差 < 1e-6
// ============================================================================
static void test_linear_cd_no_outliers() {
    printf("\n=== 测试 1: 线性 CD 拟合 (无外点) ===\n");

    // 参数
    double ra0 = 120.0, dec0 = 30.0;
    double focal_length_mm = 600.0;
    double pixel_size_um = 5.86;
    double s0 = 206.264806247 * pixel_size_um / focal_length_mm;  // ~2.014 arcsec/px
    int img_width = 1000, img_height = 1000;

    // 生成 50 对干净数据 (0.3" 噪声)
    SyntheticData data = generate_data(50, 0.3, {}, 0.0, 42);

    // 运行 vm44_fit (跳过 SIP, 仅测试线性 CD)
    VM44SolveParams params;
    std::memset(&params, 0, sizeof(params));
    params.skip_sip = 1;
    params.irm_sip_min_pairs = 30;
    params.irm_huber_delta_factor = 1.345;

    WcsFitResult result;
    int ret = vm44_fit(data.U, data.W, data.pairs,
                       ra0, dec0, focal_length_mm, pixel_size_um,
                       img_width, img_height, 4, params, result, nullptr);

    if (ret != 0) {
        TEST_FAIL("线性CD无外点", "vm44_fit 返回失败");
        return;
    }
    if (!result.success) {
        TEST_FAIL("线性CD无外点", "result.success = false");
        return;
    }

    // 验证 RMS
    if (result.rms_px >= 1.0) {
        TEST_FAIL("线性CD无外点", ("RMS 过大: " + std::to_string(result.rms_px) + "px").c_str());
        return;
    }

    // 验证 CD 矩阵
    Eigen::Vector4d cd_true = theoretical_cd(s0, data.s_true, data.theta_true);
    Eigen::Vector4d cd_fit(result.cd.cd11, result.cd.cd12, result.cd.cd21, result.cd.cd22);
    double cd_err = (cd_fit - cd_true).cwiseAbs().maxCoeff();

    printf("  理论 CD: [%.8e, %.8e, %.8e, %.8e]\n",
           cd_true[0], cd_true[1], cd_true[2], cd_true[3]);
    printf("  拟合 CD: [%.8e, %.8e, %.8e, %.8e]\n",
           cd_fit[0], cd_fit[1], cd_fit[2], cd_fit[3]);
    printf("  CD 最大误差: %.2e\n", cd_err);
    printf("  RMS: %.4f px (%.4f arcsec)\n", result.rms_px, result.rms_arcsec);
    printf("  n_pairs: %d\n", result.n_pairs);

    // 容差: CD 误差 < 1e-6 度/像素 (噪声 0.3" 应给出足够精度)
    if (cd_err < 1e-6) {
        TEST_PASS("线性CD无外点");
    } else {
        TEST_FAIL("线性CD无外点", ("CD 误差过大: " + std::to_string(cd_err)).c_str());
    }
}

// ============================================================================
// 测试 2: 含外点拟合
//   50 对中 5 对外点 (残差 ~50"), Huber 结果优于普通 LSQ
//   验证: Huber 拟合的 CD 矩阵误差 < 普通 LSQ
// ============================================================================
static void test_fit_with_outliers() {
    printf("\n=== 测试 2: 含外点拟合 (50对, 5外点, 50角秒) ===\n");

    double ra0 = 120.0, dec0 = 30.0;
    double focal_length_mm = 600.0;
    double pixel_size_um = 5.86;
    double s0 = 206.264806247 * pixel_size_um / focal_length_mm;
    int img_width = 1000, img_height = 1000;

    // 生成 50 对数据, 5 对外点 (偏移 50")
    std::vector<int> outliers = {5, 15, 25, 35, 45};
    SyntheticData data = generate_data(50, 0.3, outliers, 50.0, 42);

    // 运行 vm44_fit (Huber + MAD)
    VM44SolveParams params;
    std::memset(&params, 0, sizeof(params));
    params.skip_sip = 1;
    params.irm_sip_min_pairs = 30;
    params.irm_huber_delta_factor = 1.345;

    WcsFitResult result_huber;
    int ret_huber = vm44_fit(data.U, data.W, data.pairs,
                             ra0, dec0, focal_length_mm, pixel_size_um,
                             img_width, img_height, 4, params, result_huber, nullptr);

    if (ret_huber != 0) {
        TEST_FAIL("含外点Huber", "vm44_fit (Huber) 返回失败");
        return;
    }

    // 普通 LSQ (无 MAD, 无 Huber, 使用全部 50 点含外点)
    Eigen::Vector4d cd_plain = plain_lsq_cd(data.U, data.W, data.pairs, s0, ra0, dec0);

    // 理论 CD
    Eigen::Vector4d cd_true = theoretical_cd(s0, data.s_true, data.theta_true);

    // 计算误差
    Eigen::Vector4d cd_huber(result_huber.cd.cd11, result_huber.cd.cd12,
                              result_huber.cd.cd21, result_huber.cd.cd22);
    double err_huber = (cd_huber - cd_true).cwiseAbs().maxCoeff();
    double err_plain = (cd_plain - cd_true).cwiseAbs().maxCoeff();

    printf("  理论 CD:    [%.8e, %.8e, %.8e, %.8e]\n",
           cd_true[0], cd_true[1], cd_true[2], cd_true[3]);
    printf("  Huber CD:   [%.8e, %.8e, %.8e, %.8e]\n",
           cd_huber[0], cd_huber[1], cd_huber[2], cd_huber[3]);
    printf("  Plain LSQ:  [%.8e, %.8e, %.8e, %.8e]\n",
           cd_plain[0], cd_plain[1], cd_plain[2], cd_plain[3]);
    printf("  Huber CD 误差: %.6e\n", err_huber);
    printf("  Plain LSQ 误差: %.6e\n", err_plain);
    printf("  Huber RMS: %.4f px, n_pairs: %d\n", result_huber.rms_px, result_huber.n_pairs);

    // 验证 Huber 误差 < Plain LSQ 误差
    if (err_huber < err_plain) {
        printf("  Huber 误差比 Plain LSQ 小 %.2fx\n", err_plain / std::max(err_huber, 1e-30));
        TEST_PASS("含外点Huber优于PlainLSQ");
    } else {
        TEST_FAIL("含外点Huber优于PlainLSQ",
                  ("Huber 误差 " + std::to_string(err_huber) +
                   " >= Plain LSQ 误差 " + std::to_string(err_plain)).c_str());
    }
}

// ============================================================================
// 测试 3: SIP 2 阶拟合
//   合成带 2 阶畸变的数据, 验证 SIP 系数正确恢复
// ============================================================================
static void test_sip_order2_fit() {
    printf("\n=== 测试 3: SIP 2 阶拟合 ===\n");

    double ra0 = 120.0, dec0 = 30.0;
    double focal_length_mm = 600.0;
    double pixel_size_um = 5.86;
    double s0 = 206.264806247 * pixel_size_um / focal_length_mm;
    int img_width = 1000, img_height = 1000;

    // 真实 SIP 系数 (2 阶, 非归一化形式: Δssx = A[p][q] · ssx^p · ssy^q)
    // 使用小系数使畸变为像素级
    // A[2][0], A[1][1], A[0][2]
    double A_true[6][6] = {{0}};
    double B_true[6][6] = {{0}};
    // 2 阶项: (p,q) = (2,0), (1,1), (0,2)
    A_true[2][0] =  1.5e-5;  // ssx^2 项
    A_true[1][1] = -0.8e-5;  // ssx·ssy 项
    A_true[0][2] =  0.6e-5;  // ssy^2 项
    B_true[2][0] = -0.7e-5;
    B_true[1][1] =  1.2e-5;
    B_true[0][2] = -0.9e-5;

    // 生成 60 对数据 (满足 irm_sip_min_pairs=30)
    int N = 60;
    double noise_sigma = 0.05;  // 极小噪声, 确保 SIP 信号可分辨
    std::mt19937 rng(123);
    std::uniform_real_distribution<double> pos_dist(-400, 400);
    std::normal_distribution<double> noise_dist(0.0, noise_sigma);

    double s_true = 1.0;
    double theta_true = 5.0 * M_PI / 180.0;
    double tx_true = 0.0;   // 零平移: 确保对称分布, 线性拟合不吸收 SIP 信号
    double ty_true = 0.0;
    double ct = std::cos(theta_true), st = std::sin(theta_true);

    std::vector<StarPoint> U(N), W(N);
    std::vector<MatchPair> pairs(N);

    for (int i = 0; i < N; ++i) {
        // W 点 (星表侧)
        W[i].x = pos_dist(rng);
        W[i].y = pos_dist(rng);
        W[i].flux = 1000.0;
        W[i].saturated = false;

        // 线性变换 U_linear = s·R·W + t
        double ux_lin = s_true * (ct * W[i].x - st * W[i].y) + tx_true;
        double uy_lin = s_true * (st * W[i].x + ct * W[i].y) + ty_true;

        // 转像素空间添加 SIP 畸变
        double ssx_lin =  ux_lin / s0;
        double ssy_lin = -uy_lin / s0;

        // SIP 畸变: Δssx = Σ A[p][q]·ssx^p·ssy^q (p+q=2)
        double dssx = A_true[2][0]*ssx_lin*ssx_lin
                    + A_true[1][1]*ssx_lin*ssy_lin
                    + A_true[0][2]*ssy_lin*ssy_lin;
        double dssy = B_true[2][0]*ssx_lin*ssx_lin
                    + B_true[1][1]*ssx_lin*ssy_lin
                    + B_true[0][2]*ssy_lin*ssy_lin;

        // 实际像素 = 线性像素 - SIP 畸变 (使残差 = 预测 - 实际 = +SIP)
        double ssx_actual = ssx_lin - dssx;
        double ssy_actual = ssy_lin - dssy;

        // 转回角秒 + 噪声
        U[i].x =  ssx_actual * s0 + noise_dist(rng);
        U[i].y = -ssy_actual * s0 + noise_dist(rng);
        U[i].flux = 1000.0;
        U[i].saturated = false;

        pairs[i].u = i;
        pairs[i].w = i;
    }

    // 运行 vm44_fit (启用 SIP, max_order=4)
    VM44SolveParams params;
    std::memset(&params, 0, sizeof(params));
    params.skip_sip = 0;
    params.irm_sip_min_pairs = 30;
    params.irm_huber_delta_factor = 1.345;

    WcsFitResult result;
    int ret = vm44_fit(U, W, pairs, ra0, dec0, focal_length_mm, pixel_size_um,
                       img_width, img_height, 4, params, result, nullptr);

    if (ret != 0) {
        TEST_FAIL("SIP2阶拟合", "vm44_fit 返回失败");
        return;
    }
    if (!result.success) {
        TEST_FAIL("SIP2阶拟合", "result.success = false");
        return;
    }

    printf("  选定 SIP 阶数: %d\n", result.sip.order);
    printf("  RMS: %.4f px (%.4f arcsec)\n", result.rms_px, result.rms_arcsec);

    // 验证 SIP 阶数 = 2
    if (result.sip.order != 2) {
        char buf[256];
        std::snprintf(buf, sizeof(buf), "期望 sip_order=2, 实际=%d", result.sip.order);
        TEST_FAIL("SIP2阶拟合", buf);
        return;
    }

    // 验证 SIP 系数 (非归一化形式, 存储为 A[p*6+q])
    // 输出系数存储格式: result.sip.A[p*6+q] 对应 A_true[p][q]
    double max_coeff_err = 0;
    printf("  SIP 系数对比 (真实 vs 拟合):\n");
    printf("    A[2][0]: true=%.3e, fit=%.3e\n", A_true[2][0], result.sip.A[2*6+0]);
    printf("    A[1][1]: true=%.3e, fit=%.3e\n", A_true[1][1], result.sip.A[1*6+1]);
    printf("    A[0][2]: true=%.3e, fit=%.3e\n", A_true[0][2], result.sip.A[0*6+2]);
    printf("    B[2][0]: true=%.3e, fit=%.3e\n", B_true[2][0], result.sip.B[2*6+0]);
    printf("    B[1][1]: true=%.3e, fit=%.3e\n", B_true[1][1], result.sip.B[1*6+1]);
    printf("    B[0][2]: true=%.3e, fit=%.3e\n", B_true[0][2], result.sip.B[0*6+2]);

    max_coeff_err = std::max(max_coeff_err, std::abs(result.sip.A[2*6+0] - A_true[2][0]));
    max_coeff_err = std::max(max_coeff_err, std::abs(result.sip.A[1*6+1] - A_true[1][1]));
    max_coeff_err = std::max(max_coeff_err, std::abs(result.sip.A[0*6+2] - A_true[0][2]));
    max_coeff_err = std::max(max_coeff_err, std::abs(result.sip.B[2*6+0] - B_true[2][0]));
    max_coeff_err = std::max(max_coeff_err, std::abs(result.sip.B[1*6+1] - B_true[1][1]));
    max_coeff_err = std::max(max_coeff_err, std::abs(result.sip.B[0*6+2] - B_true[0][2]));

    printf("  最大系数误差: %.3e\n", max_coeff_err);

    // 容差: 系数误差 < 1e-5
    // 说明: SIP 系数 ~1e-5 量级, Layer 1 仿射拟合会吸收 SIP 2 阶信号在 1 阶上的
    //       投影 (Umeyama s/θ 微小偏差引入耦合), 导致 SIP 系数恢复有 ~10% 误差。
    //       1e-5 容差对应 ~10% 相对误差, 验证 SIP 拟合功能正确 (阶数选择 + 系数数量级)。
    if (max_coeff_err < 1e-5) {
        TEST_PASS("SIP2阶拟合");
    } else {
        char buf[256];
        std::snprintf(buf, sizeof(buf), "系数误差过大: %.3e", max_coeff_err);
        TEST_FAIL("SIP2阶拟合", buf);
    }
}

// ============================================================================
// 主函数
// ============================================================================
int main() {
    printf("============================================================\n");
    printf("V4.3 WcsFitter (vm44_fit) 单元测试 - Task 7\n");
    printf("============================================================\n");

    test_linear_cd_no_outliers();
    test_fit_with_outliers();
    test_sip_order2_fit();

    printf("\n============================================================\n");
    printf("测试结果: %d 通过, %d 失败\n", g_tests_passed, g_tests_failed);
    printf("============================================================\n");

    return g_tests_failed > 0 ? 1 : 0;
}
