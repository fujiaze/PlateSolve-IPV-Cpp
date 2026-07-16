// ============================================================================
// test_expand.cpp - V4.3 PairExpander 模块单元测试 (Task 3)
//
// 测试用例:
//   1. 合成数据测试: N=200, M=300 合成数据(已知 CD 变换), 验证扩充对数 ≥ 100
//   2. 自适应容差测试: 验证 τ_i 随 σ_robust 变化
//      σ_robust=2.0 → 中心 τ ≈ 6"; σ_robust=0.5 → 中心 τ ≈ 1.5"
//   3. Lowe 距离比测试: 验证 d_1/d_2 < 0.7 过滤
//   4. 双向验证测试: 验证反向投影一致性
//   5. 性能测试: N=2000, M=2000, 验证匹配耗时 < 30ms
// ============================================================================

#include "vm44_internal.h"
#include <cassert>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <vector>

// ===========================================================================
// 辅助函数
// ===========================================================================

// 构造 CD = 单位矩阵
static v44::CDMatrix make_identity_cd() {
    v44::CDMatrix cd;
    cd.cd11 = 1.0; cd.cd12 = 0.0;
    cd.cd21 = 0.0; cd.cd22 = 1.0;
    return cd;
}

// 构造 CD = scalar × I
static v44::CDMatrix make_scaled_cd(double s) {
    v44::CDMatrix cd;
    cd.cd11 = s;   cd.cd12 = 0.0;
    cd.cd21 = 0.0; cd.cd22 = s;
    return cd;
}

// 构造无 SIP (order=0)
static v44::SIPCoeffs make_no_sip() {
    v44::SIPCoeffs sip;
    std::memset(sip.A, 0, sizeof(sip.A));
    std::memset(sip.B, 0, sizeof(sip.B));
    sip.order = 0;
    return sip;
}

static v44::StarPoint make_star(double x, double y) {
    v44::StarPoint s;
    s.x = x; s.y = y; s.flux = 1.0; s.saturated = false;
    return s;
}

// 构造默认参数 (符合 vm44_types.h 中的默认值)
static v44::VM44SolveParams make_default_params() {
    v44::VM44SolveParams p;
    std::memset(&p, 0, sizeof(p));
    // PairExpander 参数
    p.region_size_px = 800;
    p.N_floor        = 5;
    p.N_cap          = 30;
    p.N_max          = 1500;
    // IRM 参数
    p.irm_tau_factor  = 3.0;
    p.irm_tau_min     = 2.0;
    p.irm_lowe_ratio  = 0.7;
    return p;
}

// ===========================================================================
// 测试 1: 合成数据测试
// N=200, M=300 合成数据(已知 CD=I 变换), 验证扩充对数 ≥ 100
// ===========================================================================
static int test_synthetic_data() {
    std::printf("[test_synthetic_data] 合成数据测试 N=200 M=300...\n");

    const int N = 200;
    std::vector<v44::StarPoint> U, W;
    v44::CDMatrix cd = make_identity_cd();
    v44::SIPCoeffs sip = make_no_sip();

    // U: 20×10 网格, 间距 30 角秒, 范围 [-300,270]×[-150,120]
    for (int i = 0; i < N; ++i) {
        double x = (i % 20) * 30.0 - 300.0;
        double y = (i / 20) * 30.0 - 150.0;
        U.push_back(make_star(x, y));
    }
    // W 前 200 个与 U 接近匹配(加 0.05 噪声)
    for (int i = 0; i < N; ++i) {
        W.push_back(make_star(U[i].x + 0.05, U[i].y - 0.05));
    }
    // W 后 100 个远离的干扰星
    for (int j = 0; j < 100; ++j) {
        W.push_back(make_star(2000.0 + j * 10.0, 2000.0 + j * 10.0));
    }

    v44::VM44SolveParams params = make_default_params();
    params.region_size_px = 100;  // 让网格分多个区域 (6×3=18 区)

    v44::ExpansionResult out;
    int ret = v44::vm44_expand(U, W, cd, sip, /*s_robust=*/1.0, /*s0=*/1.0,
                                /*ra0=*/0.0, /*dec0=*/0.0,
                                /*img_width=*/600, /*img_height=*/300,
                                params, out, nullptr);
    assert(ret == 0);
    assert(out.success);

    std::printf("  -> 扩充对数 %d (期望 ≥ 100)\n", out.n_expanded);
    assert(out.n_expanded >= 100);
    std::printf("  -> 合成数据测试  OK\n");
    return 0;
}

// ===========================================================================
// 测试 2: 自适应容差测试
// 验证 τ_i 随 σ_robust 变化
//   σ_robust=2.0 → 中心 τ ≈ 6"  (接受距离 5 的匹配)
//   σ_robust=0.5 → 中心 τ ≈ 1.5" (拒绝距离 5 的匹配)
// ===========================================================================
static int test_adaptive_tolerance() {
    std::printf("[test_adaptive_tolerance] 自适应容差测试...\n");

    v44::CDMatrix cd = make_identity_cd();
    v44::SIPCoeffs sip = make_no_sip();

    // U[0] = (0, 0) 中心位置, σ_proj = σ₀
    // W[0] = (5.0, 0.0) 与 U[0] 距离 5.0
    // W[1] = (50.0, 0.0) 远离(让 Lowe ratio 通过)
    std::vector<v44::StarPoint> U = { make_star(0.0, 0.0) };
    std::vector<v44::StarPoint> W = { make_star(5.0, 0.0), make_star(50.0, 0.0) };

    v44::VM44SolveParams params = make_default_params();
    params.irm_tau_min = 0.0;  // 让 τ 完全由 sigma 决定

    // 场景 A: σ_robust=2.0 → σ₀=2.0, τ_中心=6.0, 5.0<6.0 → 接受
    {
        v44::ExpansionResult out;
        int ret = v44::vm44_expand(U, W, cd, sip, /*s_robust=*/2.0, /*s0=*/1.0,
                                    0.0, 0.0, 600, 300, params, out, nullptr);
        assert(ret == 0);
        std::printf("  -> σ_robust=2.0: 扩充 %d 对 (期望 1, τ_中心≈6\")\n",
                    out.n_expanded);
        assert(out.n_expanded == 1);
        assert(out.candidates[0].u == 0);
        assert(out.candidates[0].w == 0);
    }

    // 场景 B: σ_robust=0.5 → σ₀=0.5, τ_中心=1.5, 5.0>1.5 → 拒绝
    {
        v44::ExpansionResult out;
        int ret = v44::vm44_expand(U, W, cd, sip, /*s_robust=*/0.5, /*s0=*/1.0,
                                    0.0, 0.0, 600, 300, params, out, nullptr);
        assert(ret == 0);
        std::printf("  -> σ_robust=0.5: 扩充 %d 对 (期望 0, τ_中心≈1.5\")\n",
                    out.n_expanded);
        assert(out.n_expanded == 0);
    }

    std::printf("  -> 自适应容差测试  OK\n");
    return 0;
}

// ===========================================================================
// 测试 3: Lowe 距离比测试
// 验证 d_1/d_2 < 0.7 过滤
//   场景 A: W[0]=(1,0), W[1]=(2,0) → ratio=0.5<0.7 → 接受
//   场景 B: W[0]=(1,0), W[1]=(1.1,0) → ratio=0.91>0.7 → 拒绝
// ===========================================================================
static int test_lowe_ratio() {
    std::printf("[test_lowe_ratio] Lowe 距离比测试...\n");

    v44::CDMatrix cd = make_identity_cd();
    v44::SIPCoeffs sip = make_no_sip();
    std::vector<v44::StarPoint> U = { make_star(0.0, 0.0) };

    v44::VM44SolveParams params = make_default_params();
    params.irm_tau_min = 100.0;  // 让 τ 截断不生效, 只测 Lowe

    // 场景 A: ratio = 1.0/2.0 = 0.5 < 0.7 → 接受
    {
        std::vector<v44::StarPoint> W = { make_star(1.0, 0.0), make_star(2.0, 0.0) };
        v44::ExpansionResult out;
        int ret = v44::vm44_expand(U, W, cd, sip, /*s_robust=*/10.0, /*s0=*/1.0,
                                    0.0, 0.0, 600, 300, params, out, nullptr);
        assert(ret == 0);
        std::printf("  -> 场景A (ratio=0.5): 扩充 %d 对 (期望 1)\n", out.n_expanded);
        assert(out.n_expanded == 1);
        assert(out.candidates[0].w == 0);
    }

    // 场景 B: ratio = 1.0/1.1 = 0.909 > 0.7 → 拒绝
    {
        std::vector<v44::StarPoint> W = { make_star(1.0, 0.0), make_star(1.1, 0.0) };
        v44::ExpansionResult out;
        int ret = v44::vm44_expand(U, W, cd, sip, /*s_robust=*/10.0, /*s0=*/1.0,
                                    0.0, 0.0, 600, 300, params, out, nullptr);
        assert(ret == 0);
        std::printf("  -> 场景B (ratio=0.91): 扩充 %d 对 (期望 0)\n", out.n_expanded);
        assert(out.n_expanded == 0);
    }

    std::printf("  -> Lowe 距离比测试  OK\n");
    return 0;
}

// ===========================================================================
// 测试 4: 双向验证测试
// CD = 10×I (x,y 方向都放大 10 倍)
// U[0]=(0,0), U[1]=(1,0)
// W[0]=(6,0), W[1]=(100,0) (干扰星让 Lowe 通过)
// 投影: proj_0=(0,0) 离 W[0] 距离 6 (Lowe 通过, τ 通过)
//   反向: proj_back_0=(0.6,0), 最近邻是 U[1](距离 0.4) ≠ U[0] → 拒绝
// 投影: proj_1=(10,0) 离 W[0] 距离 4 (Lowe 通过, τ 通过)
//   反向: proj_back_for_W[0]=(0.6,0), 最近邻是 U[1](距离 0.4) = U[1] → 保留
// 期望: 最终匹配含 U[1]↔W[0], 不含 U[0]↔W[0]
// ===========================================================================
static int test_bidirectional_verification() {
    std::printf("[test_bidirectional_verification] 双向验证测试...\n");

    v44::CDMatrix cd = make_scaled_cd(10.0);  // CD = 10×I
    v44::SIPCoeffs sip = make_no_sip();

    std::vector<v44::StarPoint> U = {
        make_star(0.0, 0.0),  // U[0]
        make_star(1.0, 0.0)   // U[1]
    };
    std::vector<v44::StarPoint> W = {
        make_star(6.0, 0.0),    // W[0]
        make_star(100.0, 0.0)   // W[1] (干扰星)
    };

    v44::VM44SolveParams params = make_default_params();
    // s_robust=3.0 → σ₀=3.0, τ_中心=9.0, 让 6<9 和 4<9 都通过 τ
    // irm_tau_min 默认 2.0, 不会限制
    v44::ExpansionResult out;
    int ret = v44::vm44_expand(U, W, cd, sip, /*s_robust=*/3.0, /*s0=*/1.0,
                                0.0, 0.0, 600, 300, params, out, nullptr);
    assert(ret == 0);

    std::printf("  -> 扩充 %d 对\n", out.n_expanded);

    // 检查 U[0]↔W[0] 被拒绝, U[1]↔W[0] 被保留
    bool has_u0 = false, has_u1 = false;
    for (const auto& mp : out.candidates) {
        std::printf("    匹配对: U[%d] ↔ W[%d]\n", mp.u, mp.w);
        if (mp.u == 0) has_u0 = true;
        if (mp.u == 1 && mp.w == 0) has_u1 = true;
    }
    assert(!has_u0);  // U[0] 应被双向验证拒绝
    assert(has_u1);   // U[1]↔W[0] 应保留
    std::printf("  -> 双向验证测试  OK (U[0] 被拒, U[1] 保留)\n");
    return 0;
}

// ===========================================================================
// 测试 5: 性能测试
// N=2000, M=2000, 验证匹配耗时 < 30ms
// ===========================================================================
static int test_performance() {
    std::printf("[test_performance] 性能测试 N=2000 M=2000...\n");

    const int N = 2000;
    std::vector<v44::StarPoint> U, W;
    v44::CDMatrix cd = make_identity_cd();
    v44::SIPCoeffs sip = make_no_sip();

    // U: 网格分布, 50×40=2000, 间距 60 角秒
    // 范围 [-1500, 1440]×[-1200, 1140]
    for (int i = 0; i < N; ++i) {
        double x = (i % 50) * 60.0 - 1500.0;
        double y = (i / 50) * 60.0 - 1200.0;
        U.push_back(make_star(x, y));
    }
    // W: 前 2000 个与 U 接近匹配(加 0.01 噪声)
    for (int i = 0; i < N; ++i) {
        W.push_back(make_star(U[i].x + 0.01, U[i].y - 0.01));
    }

    v44::VM44SolveParams params = make_default_params();

    auto t0 = std::chrono::high_resolution_clock::now();
    v44::ExpansionResult out;
    int ret = v44::vm44_expand(U, W, cd, sip, /*s_robust=*/1.0, /*s0=*/1.0,
                                0.0, 0.0, 3000, 2400, params, out, nullptr);
    auto t1 = std::chrono::high_resolution_clock::now();
    double ms = std::chrono::duration<double, std::milli>(t1 - t0).count();

    assert(ret == 0);
    std::printf("  -> 扩充 %d 对, 耗时 %.2f ms (期望 < 30ms)\n",
                out.n_expanded, ms);
    assert(ms < 30.0);
    std::printf("  -> 性能测试  OK\n");
    return 0;
}

// ===========================================================================
// 主函数
// ===========================================================================
int main() {
    std::printf("=== test_expand 开始 ===\n");
    int failed = 0;
    failed += test_synthetic_data();
    failed += test_adaptive_tolerance();
    failed += test_lowe_ratio();
    failed += test_bidirectional_verification();
    failed += test_performance();
    if (failed == 0) {
        std::printf("=== test_expand 全部通过 ===\n");
        return 0;
    } else {
        std::printf("=== test_expand 失败 %d 项 ===\n", failed);
        return 1;
    }
}
