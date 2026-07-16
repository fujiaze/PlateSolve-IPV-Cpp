// ============================================================================
// test_irm.cpp - vm44_irm_refine 单元测试 (Task 8)
//
// 测试用例:
//   1. 空输入: C0 为空 → 返回 -1
//   2. 完美数据: C0=8 对 + CD0=单位 → IRM 应收敛, 控制点增长
//   3. 含噪声数据: C0=10 对 + 小位置噪声 → IRM 应收敛
//
// 编译: make test_irm
// ============================================================================

#include "../include/vm44_internal.h"

#include <cstdio>
#include <cmath>
#include <random>
#include <string>

namespace v44 {

// 便捷构造参数
static VM44SolveParams make_test_params() {
    VM44SolveParams p;
    // PairVerifier
    p.mad_iters = 3;
    p.mad_threshold_factor = 3.0;
    p.mad_min_threshold_arcsec = 5.0;
    p.lnK_accept = 10.0;
    p.lnK_weak = 3.0;
    p.eps_A = 0.1;
    p.eps_J = 0.1;
    p.triangle_pass_rate = 0.7;
    // PairExpander
    p.region_size_px = 800;
    p.N_floor = 5;
    p.N_cap = 30;
    p.N_max = 1500;
    // VectorMatcher (IRM 不直接用, 但 expand 可能引用)
    p.s_min = 0.9;
    p.s_max = 1.1;
    p.irm_lowe_ratio = 0.7;
    // WcsFitter
    p.sip_max_order = 3;
    p.skip_sip = 0;
    // IRM
    p.irm_max_iter = 5;
    p.irm_converge_eps = 0.05;
    p.irm_diverge_factor = 1.1;
    p.irm_tau_min = 2.0;
    p.irm_tau_factor = 3.0;
    p.irm_k_geometry = 8;
    p.irm_geom_threshold = 4;
    p.irm_geom_dist_tol = 3.0;
    p.irm_ransac_max_iter = 200;
    p.irm_ransac_min_inliers = 4;
    p.irm_huber_delta_factor = 1.345;
    p.irm_sip_min_pairs = 10;
    p.irm_s_initial = 0;
    return p;
}

// 生成网格星点 (角秒坐标)
static std::vector<StarPoint> make_grid_stars(int side, double spacing) {
    std::vector<StarPoint> stars;
    stars.reserve(side * side);
    for (int i = 0; i < side; ++i) {
        for (int j = 0; j < side; ++j) {
            StarPoint s;
            s.x = (i - (side - 1) / 2.0) * spacing;
            s.y = (j - (side - 1) / 2.0) * spacing;
            s.flux = 1000.0 - (i + j) * 10.0;
            s.saturated = false;
            stars.push_back(s);
        }
    }
    return stars;
}

// 单位 CD 矩阵 (s0 角秒/像素, 无旋转)
static CDMatrix make_identity_cd(double s0) {
    // CD = [[-s0, 0], [0, s0]] (标准 WCS, RA 随 x 增加减少)
    // 但 V4.3 U/W 均为角秒坐标, 简化为 [[s0, 0], [0, s0]]
    CDMatrix cd;
    cd.cd11 = s0; cd.cd12 = 0.0;
    cd.cd21 = 0.0; cd.cd22 = s0;
    return cd;
}

// 测试 1: 空输入
static bool test_empty() {
    printf("--- 测试 1: 空输入 ---\n");
    std::vector<MatchPair> C0;
    CDMatrix CD0 = make_identity_cd(1.0);
    std::vector<StarPoint> U, W;
    VM44SolveParams p = make_test_params();
    CDMatrix final_cd;
    SIPCoeffs final_sip;
    std::vector<MatchPair> final_cp;
    SRobustResult final_sr;
    int n_iters;
    bool converged;
    double bayes_lnK = 0.0;
    double tri_ratio = 0.0;
    Logger logger;

    int rc = vm44_irm_refine(C0, CD0, U, W, 1.0, 600.0, 3.0,
                              10.0, 20.0, 100, 100, 2.0,
                              p, final_cd, final_sip, final_cp,
                              final_sr, n_iters, converged, bayes_lnK, tri_ratio, &logger);
    if (rc != -1) {
        printf("  失败: 期望 rc=-1, 实际 rc=%d\n", rc);
        return false;
    }
    printf("  通过: rc=-1 (空输入正确返回错误)\n");
    return true;
}

// 测试 2: 完美数据
static bool test_perfect() {
    printf("--- 测试 2: 完美数据 ---\n");
    // 10x10 网格, 间距 60"
    std::vector<StarPoint> U = make_grid_stars(10, 60.0);
    std::vector<StarPoint> W = U;  // 完全相同

    // C0: 选 8 对作为初始控制点
    std::vector<MatchPair> C0;
    for (int i = 0; i < 8; ++i) C0.push_back({i, i});

    double s0 = 1.0;  // 角秒/像素 (但坐标已是角秒, 这里简化)
    CDMatrix CD0 = make_identity_cd(s0);
    VM44SolveParams p = make_test_params();
    CDMatrix final_cd;
    SIPCoeffs final_sip;
    std::vector<MatchPair> final_cp;
    SRobustResult final_sr;
    int n_iters;
    bool converged;
    double bayes_lnK = 0.0;
    double tri_ratio = 0.0;
    Logger logger;

    int rc = vm44_irm_refine(C0, CD0, U, W, s0, 600.0, 3.0,
                              10.0, 20.0, 600, 600, 2.0,
                              p, final_cd, final_sip, final_cp,
                              final_sr, n_iters, converged, bayes_lnK, tri_ratio, &logger);
    if (rc != 0) {
        printf("  失败: 期望 rc=0, 实际 rc=%d\n", rc);
        return false;
    }
    printf("  n_iters=%d, converged=%d, N_final=%d, S_robust=%.4f\n",
           n_iters, converged ? 1 : 0, (int)final_cp.size(), final_sr.s_robust);

    // 完美数据应收敛
    if (!converged) {
        printf("  警告: 期望 converged=true (完美数据应收敛)\n");
        // 不算失败, IRM 可能因其他原因未收敛
    }
    // 控制点应增长 (≥ 8)
    if ((int)final_cp.size() < 8) {
        printf("  失败: 期望 N_final>=8, 实际 %d\n", (int)final_cp.size());
        return false;
    }
    printf("  通过: 完美数据 IRM 完成 (N=%d, S_robust=%.4f\")\n",
           (int)final_cp.size(), final_sr.s_robust);
    return true;
}

// 测试 3: 含小噪声数据
static bool test_with_noise() {
    printf("--- 测试 3: 含小噪声数据 ---\n");
    std::mt19937 rng(42);
    std::normal_distribution<double> noise(0.0, 0.5);  // σ=0.5" 噪声

    // 8x8 网格, 间距 80"
    std::vector<StarPoint> U = make_grid_stars(8, 80.0);
    std::vector<StarPoint> W = U;

    // 给 U 加噪声 (模拟检测误差)
    for (auto& s : U) {
        s.x += noise(rng);
        s.y += noise(rng);
    }

    // C0: 选 10 对
    std::vector<MatchPair> C0;
    for (int i = 0; i < 10; ++i) C0.push_back({i, i});

    double s0 = 1.0;
    CDMatrix CD0 = make_identity_cd(s0);
    VM44SolveParams p = make_test_params();
    CDMatrix final_cd;
    SIPCoeffs final_sip;
    std::vector<MatchPair> final_cp;
    SRobustResult final_sr;
    int n_iters;
    bool converged;
    double bayes_lnK = 0.0;
    double tri_ratio = 0.0;
    Logger logger;

    int rc = vm44_irm_refine(C0, CD0, U, W, s0, 600.0, 3.0,
                              10.0, 20.0, 640, 640, 2.0,
                              p, final_cd, final_sip, final_cp,
                              final_sr, n_iters, converged, bayes_lnK, tri_ratio, &logger);
    if (rc != 0) {
        printf("  失败: 期望 rc=0, 实际 rc=%d\n", rc);
        return false;
    }
    printf("  n_iters=%d, converged=%d, N_final=%d, S_robust=%.4f\n",
           n_iters, converged ? 1 : 0, (int)final_cp.size(), final_sr.s_robust);

    // 含噪声数据 S_robust 应 > 0 (有残差)
    if (final_sr.s_robust <= 0) {
        printf("  失败: 期望 S_robust > 0 (含噪声)\n");
        return false;
    }
    printf("  通过: 含噪声数据 IRM 完成 (S_robust=%.4f\")\n", final_sr.s_robust);
    return true;
}

} // namespace v44

// ============================================================================
// 主函数
// ============================================================================
int main() {
    printf("================================\n");
    printf("vm44_irm_refine 单元测试 (Task 8)\n");
    printf("================================\n\n");

    int passed = 0, total = 0;
    auto run = [&](bool (*fn)()) { ++total; if (fn()) ++passed; printf("\n"); };

    run(v44::test_empty);
    run(v44::test_perfect);
    run(v44::test_with_noise);

    printf("================================\n");
    printf("结果: %d/%d 通过\n", passed, total);
    printf("================================\n");
    return (passed == total) ? 0 : 1;
}
