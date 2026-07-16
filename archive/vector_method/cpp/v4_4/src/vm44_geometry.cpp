// ============================================================================
// vm44_geometry.cpp - V4.3 局部几何一致性过滤模块 (IRM Step 2)
//
// 职责: K=8 邻星角距一致性过滤, 借鉴 Heyl 2013 k-d match 思想
// 对每个候选匹配 (img_A, gaia_a), 检查图像侧邻星与星表侧邻星的角距
// 是否一致, 一致性计数 < 阈值 (默认 4/8) 则丢弃该候选
//
// 说明: U 与 W 均为角秒坐标 (原点在图像中心), 可直接用欧氏距离计算角距
//
// Task 4 实现
// ============================================================================

#include "vm44_internal.h"
#include <algorithm>
#include <cmath>
#include <chrono>
#include <utility>
#include <vector>

namespace v44 {

namespace {

// 两点间欧氏距离平方 (角秒坐标)
inline double dist2(double x1, double y1, double x2, double y2) {
    double dx = x1 - x2, dy = y1 - y2;
    return dx * dx + dy * dy;
}

// K 近邻线性扫描 (N<=2000 无需 KDTree)
// points  : 点数组
// qx,qy   : 查询点坐标
// K       : 邻居数
// exclude : 需排除的索引 (通常为查询点自身, -1 表示不排除)
// 返回    : K 个最近邻索引, 按距离升序
std::vector<int> knn(const std::vector<StarPoint>& points,
                     double qx, double qy, int K, int exclude) {
    std::vector<std::pair<double, int>> dists;
    dists.reserve(points.size());
    for (int i = 0; i < (int)points.size(); ++i) {
        if (i == exclude) continue;
        dists.emplace_back(dist2(qx, qy, points[i].x, points[i].y), i);
    }
    int K_eff = std::min(K, (int)dists.size());
    if (K_eff <= 0) return {};
    // nth_element 取前 K, 再排序保证升序
    std::nth_element(dists.begin(), dists.begin() + K_eff, dists.end(),
        [](const std::pair<double, int>& a, const std::pair<double, int>& b) {
            return a.first < b.first;
        });
    std::sort(dists.begin(), dists.begin() + K_eff,
        [](const std::pair<double, int>& a, const std::pair<double, int>& b) {
            return a.first < b.first;
        });
    std::vector<int> result;
    result.reserve(K_eff);
    for (int i = 0; i < K_eff; ++i) result.push_back(dists[i].second);
    return result;
}

} // anonymous namespace

int vm44_geometry_filter(
    const std::vector<MatchPair>& candidates,
    const std::vector<StarPoint>& U,
    const std::vector<StarPoint>& W,
    double s0,
    double s_robust,
    const VM44SolveParams& params,
    std::vector<MatchPair>& filtered,
    Logger* logger)
{
    filtered.clear();

    if (candidates.empty()) {
        if (logger) logger->warn("vm44_geometry_filter: 候选为空, 无需过滤");
        return 0;
    }
    if (U.empty() || W.empty()) {
        if (logger) logger->error("vm44_geometry_filter: U 或 W 为空");
        return -1;
    }

    // 参数: 图像侧邻星数用参数, 星表侧固定 15, 只检查最近 5 个
    const int K_img         = params.irm_k_geometry;        // 默认 8
    const int K_gaia        = 15;
    const int K_gaia_check  = 5;
    const int threshold     = params.irm_geom_threshold;    // 默认 4
    // 角距容差: max(tol, tol × S_robust), tol 默认 3.0
    const double tol = std::max(params.irm_geom_dist_tol,
                                params.irm_geom_dist_tol * s_robust);

    if (logger) {
        logger->infof("vm44_geometry_filter: 候选=%d, K_img=%d, K_gaia=%d, "
                      "threshold=%d, tol=%.3f, s0=%.4f, s_robust=%.4f",
                      (int)candidates.size(), K_img, K_gaia, threshold,
                      tol, s0, s_robust);
    }

    auto t0 = std::chrono::steady_clock::now();
    int n_kept = 0, n_dropped = 0;
    filtered.reserve(candidates.size());

    for (int ci = 0; ci < (int)candidates.size(); ++ci) {
        const MatchPair& mp = candidates[ci];
        if (mp.u < 0 || mp.u >= (int)U.size() ||
            mp.w < 0 || mp.w >= (int)W.size()) {
            if (logger) logger->warnf("vm44_geometry_filter: 候选 %d 索引越界 (u=%d,w=%d)",
                                      ci, mp.u, mp.w);
            continue;
        }

        const StarPoint& img_A  = U[mp.u];
        const StarPoint& gaia_a = W[mp.w];

        // 图像侧 K 近邻 (排除 img_A 自身)
        std::vector<int> nb_img  = knn(U, img_A.x, img_A.y, K_img, mp.u);
        // 星表侧 K 近邻 (排除 gaia_a 自身)
        std::vector<int> nb_gaia = knn(W, gaia_a.x, gaia_a.y, K_gaia, mp.w);

        int consistency = 0;
        for (int bi = 0; bi < (int)nb_img.size(); ++bi) {
            const StarPoint& img_B = U[nb_img[bi]];
            // 图像侧角距: U 已是角秒坐标 (Y 向上, 原点在中心), 与 W 同单位
            // 直接用欧氏距离即为角距, 无需 × s0
            double d_img = std::sqrt(dist2(img_A.x, img_A.y, img_B.x, img_B.y));

            // 在星表侧最近 K_gaia_check 个邻居中查找角距一致的 gaia_b
            bool matched = false;
            int check_n = std::min(K_gaia_check, (int)nb_gaia.size());
            for (int gj = 0; gj < check_n; ++gj) {
                const StarPoint& gaia_b = W[nb_gaia[gj]];
                double d_gaia = std::sqrt(dist2(gaia_a.x, gaia_a.y, gaia_b.x, gaia_b.y));
                if (std::fabs(d_img - d_gaia) < tol) {
                    matched = true;
                    break;
                }
            }
            if (matched) ++consistency;
        }

        if (consistency >= threshold) {
            filtered.push_back(mp);
            ++n_kept;
        } else {
            ++n_dropped;
        }
    }

    auto t1 = std::chrono::steady_clock::now();
    double ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
    if (logger) {
        logger->infof("vm44_geometry_filter: 保留=%d, 丢弃=%d, 耗时=%.2f ms",
                      n_kept, n_dropped, ms);
    }

    return 0;
}

} // namespace v44
