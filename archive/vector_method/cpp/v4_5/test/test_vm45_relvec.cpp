// ============================================================================
// test_vm45_relvec.cpp - V4.5 Phase A 合成数据单元测试
//
// 生成 N_W=80 颗 W (Gaia 侧, 角秒, 范围 ±500") + 已知相似变换
//   U = s·R(θ)·W + t  (s=1, θ=30°, tx=500", ty=300")
// 取前 N_U=50 颗 W 经变换得到 U (无位置噪声, 严格按 spec.md "无噪声" 场景)
// → 调用 vm45_relvec_match()
// 断言:
//   1. rc == 0 (函数成功)
//   2. |θ_peak - (-θ_true)| < 0.5° (含 180° 模糊性 wrap)
//      算法定义 Δθ = angle_gaia - angle_img, 对 U=R(θ)W+t 有 Δθ = -θ
//   3. SNR > 10.0
//   4. n_passed > 0
//
// 编译 (推荐: 静态编译 vm45_relvec.cpp, 避免运行时 DLL 依赖):
//   g++ -O2 -std=c++17 -Iinclude \
//       test/test_vm45_relvec.cpp src/vm45_relvec.cpp \
//       -o test/test_vm45_relvec.exe
//
// 运行:
//   ./test/test_vm45_relvec.exe
//
// 注: vm45_relvec_match 是 namespace v45 内的 C++ 函数 (非 extern "C"),
//     测试源文件直接 #include "vm45_internal.h" 后调用即可, 不需 DLL 导出。
//     vm45_select.cpp / vm45_entry.cpp 均不需要 (测试不调用 vm45_solve / vm45_select)。
// ============================================================================

#include "../include/vm45_internal.h"
#include "../include/vm45_log.h"

#include <iostream>
#include <vector>
#include <cmath>
#include <cstring>
#include <random>
#include <cstdio>
#include <chrono>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

using namespace v45;

// ============================================================================
// 合成数据生成
// ============================================================================

// 生成 N 颗 W 星 (Gaia 侧), 均匀分布在 [-500, 500]" x [-500, 500]" 范围
// FOV 对角线 ≈ sqrt(1000² + 1000²) ≈ 1414" (与 spec "FOV~1000" 范围" 一致)
static std::vector<StarPoint> generate_w_stars(int N, std::mt19937& rng) {
    std::uniform_real_distribution<double> uni(-500.0, 500.0);
    std::vector<StarPoint> W(N);
    for (int i = 0; i < N; ++i) {
        W[i].x = uni(rng);
        W[i].y = uni(rng);
        W[i].flux = 1000.0f;
        W[i].saturated = false;
    }
    return W;
}

// 应用相似变换: U = s·R(θ)·W + t, 然后加高斯位置噪声
// 取前 M 颗 W (M ≤ N_W), 经变换得到 M 颗 U
//
// 旋转矩阵 R(θ) (CCW by θ):
//   U.x = s·(cos θ · W.x - sin θ · W.y) + tx
//   U.y = s·(sin θ · W.x + cos θ · W.y) + ty
static std::vector<StarPoint> apply_transform(
    const std::vector<StarPoint>& W,
    int M, double s, double theta_deg, double tx, double ty,
    double noise_sigma, std::mt19937& rng)
{
    std::normal_distribution<double> noise(0.0, noise_sigma);
    double theta_rad = theta_deg * M_PI / 180.0;
    double ct = std::cos(theta_rad), st = std::sin(theta_rad);

    int N = (int)W.size();
    if (M > N) M = N;
    std::vector<StarPoint> U(M);
    for (int i = 0; i < M; ++i) {
        double wx = W[i].x, wy = W[i].y;
        double rx = s * (ct * wx - st * wy);
        double ry = s * (st * wx + ct * wy);
        U[i].x = rx + tx + noise(rng);
        U[i].y = ry + ty + noise(rng);
        U[i].flux = W[i].flux;
        U[i].saturated = W[i].saturated;
    }
    return U;
}

// ============================================================================
// 辅助: wrap 角度差到 [-180, 180]
// ============================================================================
static double wrap180(double x) {
    double r = std::fmod(x, 360.0);
    if (r >= 180.0)  r -= 360.0;
    if (r < -180.0)  r += 360.0;
    return r;
}

// ============================================================================
// 主函数
// ============================================================================
int main(int argc, char** argv) {
    std::cout << "=== V4.5 Phase A RelVec 合成数据测试 ===" << std::endl;
    std::cout << std::endl;

    // ------------------------------------------------------------------
    // 测试参数 (严格按 spec.md: θ=30°, s=1.0, t=(500", 300"), 无噪声)
    // ------------------------------------------------------------------
    const int    N_W          = 80;     // Gaia 侧星数 (W)
    const int    N_U          = 50;     // 图像侧星数 (U, 取 W 前 50 颗变换)
    const double s_true       = 1.0;    // 缩放因子
    const double theta_true   = 30.0;   // 旋转角 (度)
    const double tx_true      = 500.0;  // X 平移 (角秒)
    const double ty_true      = 300.0;  // Y 平移 (角秒)
    const double noise_sigma  = 0.0;    // 位置噪声 σ (角秒, spec 要求 "无噪声")

    // 固定随机种子 (可复现)
    const unsigned seed = 42;
    std::mt19937 rng(seed);

    // ------------------------------------------------------------------
    // Step 1: 生成 W (Gaia 侧, 80 颗, 均匀分布 ±500")
    // ------------------------------------------------------------------
    auto W = generate_w_stars(N_W, rng);
    std::cout << "[Step 1] 生成 W: " << N_W << " 颗, 范围 [-500, 500]\" x [-500, 500]\""
              << std::endl;

    // ------------------------------------------------------------------
    // Step 2: 生成 U (取前 N_U=50 颗 W, 经 s·R(θ)·W + t 变换, 无噪声)
    // ------------------------------------------------------------------
    auto U = apply_transform(W, N_U, s_true, theta_true, tx_true, ty_true,
                              noise_sigma, rng);
    std::cout << "[Step 2] 生成 U: " << N_U << " 颗"
              << ", s=" << s_true
              << ", θ=" << theta_true << "°"
              << ", t=(" << tx_true << "\", " << ty_true << "\")"
              << ", 噪声 σ=" << noise_sigma << "\""
              << std::endl;

    // ------------------------------------------------------------------
    // Step 3: 手动初始化 VM45SolveParams
    // (避免依赖 vm45_entry.cpp 中的 vm45_get_default_params, 测试只链接 vm45_relvec.cpp)
    // 默认值与 vm45_entry.cpp::vm45_get_default_params 一致
    // ------------------------------------------------------------------
    VM45SolveParams params;
    std::memset(&params, 0, sizeof(params));

    // 基础
    params.seed = 42;

    // Phase 0 (StarSelector, 测试不调用, 但填充默认值以保完整)
    params.img_n_target              = 50;
    params.gaia_density_ratio        = 1.5;
    params.gaia_query_radius_factor  = 0.55;
    params.m_lim_step                = 0.5;
    params.m_lim_max_iter            = 10;
    params.density_tolerance         = 0.1;

    // Phase A (相对向量法, 严格按设计文档)
    params.K_total                   = 20000;  // 总采样次数
    params.sigma_d_px                = 2.0;    // 距离容差 σ_d (像素)
    params.n_third                   = 0;      // 0=用全部可用第三星
    params.third_ratio_min           = 0.3;    // 第三星通过比例阈值
    params.theta_bw                  = 1.0;    // θ 直方图 bin 宽度 (度)
    params.snr_threshold             = 5.0;    // SNR 接受阈值
    params.relvec_max_u              = 100;    // U 组限流上限
    params.relvec_max_cand           = 500;    // 单次采样候选对上限
    params.relvec_min_len_frac       = 0.05;   // 最小星对距离比例
    params.relvec_max_len_frac       = 0.8;    // 最大星对距离比例

    // 自适应采样停止
    params.adaptive_stop             = 1;
    params.min_samples               = 200;
    params.check_interval            = 100;
    params.snr_eps                   = 0.05;
    params.max_stable                = 3;

    // 日志 (NULL = 不写 CSV, 但 Logger 仍输出到 stderr)
    params.log_dir                   = nullptr;

    std::cout << "[Step 3] 参数初始化: K_total=" << params.K_total
              << ", σ_d_px=" << params.sigma_d_px << "px"
              << ", n_third=" << params.n_third
              << ", ratio_min=" << params.third_ratio_min
              << ", θ_bw=" << params.theta_bw << "°"
              << ", SNR_thresh=" << params.snr_threshold
              << std::endl;
    std::cout << std::endl;

    // ------------------------------------------------------------------
    // Step 4: 调用 vm45_relvec_match
    // ------------------------------------------------------------------
    Logger logger;  // 未 init 文件, 仅输出到 stderr
    RelVecResult result;
    int rc = 0;

    auto t0 = std::chrono::steady_clock::now();
    rc = vm45_relvec_match(U, W, s_true, params, result, &logger);
    auto t1 = std::chrono::steady_clock::now();
    double t_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();

    std::cout << "[Step 4] vm45_relvec_match 完成, 耗时 " << t_ms << " ms" << std::endl;
    std::cout << "  rc           = " << rc << std::endl;
    std::cout << "  success      = " << (result.success ? "true" : "false") << std::endl;
    std::cout << "  θ_peak       = " << result.theta_peak_deg << "°" << std::endl;
    std::cout << "  SNR          = " << result.theta_snr << std::endl;
    std::cout << "  peak_bin     = " << result.peak_bin << std::endl;
    std::cout << "  bg_median    = " << result.bg_median << std::endl;
    std::cout << "  n_passed     = " << result.n_passed << std::endl;
    std::cout << "  n_samples    = " << result.n_samples << std::endl;
    std::cout << "  n_total_cand = " << result.n_total_candidates << std::endl;
    std::cout << "  passed_pairs = " << result.passed_pairs.size() << std::endl;
    std::cout << std::endl;

    // ------------------------------------------------------------------
    // Step 5: 断言
    // ------------------------------------------------------------------
    bool ok = true;

    // 断言 1: rc == 0 (函数成功返回)
    if (rc != 0) {
        std::cout << "[FAIL] 断言 1: vm45_relvec_match 返回非 0 (rc=" << rc << ")"
                  << std::endl;
        ok = false;
    } else {
        std::cout << "[PASS] 断言 1: rc == 0" << std::endl;
    }

    // 断言 2: θ 误差 < 0.5°
    //
    // 算法定义: Δθ = angle_gaia - angle_img
    // 对于变换 U = R(θ)·W + t:
    //   ΔU = R(θ)·ΔW
    //   angle_img = angle(ΔU) = angle(ΔW) + θ = angle_gaia + θ
    //   Δθ = angle_gaia - angle_img = -θ
    // 故期望 θ_peak ≈ -θ_true = -30°
    //
    // 又因采样 (i,j) 与 (j,i) 各一半概率, 产生两个相距 180° 的峰:
    //   主峰: -θ_true = -30°
    //   次峰: -θ_true + 180° = 150°
    // 故测试需接受任一峰, 即 θ_peak 与 -θ_true 的差 (mod 180°) < 0.5°
    const double theta_expected = -theta_true;  // -30°
    double diff = wrap180(result.theta_peak_deg - theta_expected);
    double abs_diff = std::abs(diff);
    // 180° 模糊: 取 |diff| 与 |180 - |diff|| 的较小值
    double theta_err = std::min(abs_diff, std::abs(180.0 - abs_diff));

    if (theta_err > 0.5) {
        std::cout << "[FAIL] 断言 2: θ 误差 " << theta_err << "° > 0.5°"
                  << " (θ_peak=" << result.theta_peak_deg << "°"
                  << ", 期望=" << theta_expected << "° 或 " << (theta_expected + 180.0) << "°)"
                  << std::endl;
        ok = false;
    } else {
        std::cout << "[PASS] 断言 2: θ 误差 " << theta_err << "° <= 0.5° (θ_bw 离散化极限)"
                  << " (θ_peak=" << result.theta_peak_deg << "°"
                  << ", 期望=" << theta_expected << "°)"
                  << std::endl;
    }

    // 断言 3: SNR > 10.0
    if (result.theta_snr <= 10.0) {
        std::cout << "[FAIL] 断言 3: SNR " << result.theta_snr << " <= 10.0" << std::endl;
        ok = false;
    } else {
        std::cout << "[PASS] 断言 3: SNR " << result.theta_snr << " > 10.0" << std::endl;
    }

    // 断言 4: n_passed > 0
    if (result.n_passed <= 0) {
        std::cout << "[FAIL] 断言 4: n_passed " << result.n_passed << " <= 0" << std::endl;
        ok = false;
    } else {
        std::cout << "[PASS] 断言 4: n_passed " << result.n_passed << " > 0" << std::endl;
    }

    std::cout << std::endl;
    if (ok) {
        std::cout << "=== 所有断言通过 ===" << std::endl;
        return 0;
    } else {
        std::cout << "=== 测试失败 ===" << std::endl;
        return 1;
    }
}
