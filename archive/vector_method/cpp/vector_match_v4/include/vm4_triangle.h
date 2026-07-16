#ifndef VM4_TRIANGLE_H
#define VM4_TRIANGLE_H

// ============================================================================
// vm4_triangle.h - V4.0 三角形双特征二级验证（Task 6）
//
// Phase D' 二级验证：对匹配对集合做三角形几何一致性检验
//   特征 1: 三角形面积 A（海伦公式）
//   特征 2: 极惯性矩 J = A·(a²+b²+c²)/36
//
// 参考: Cole 2006（三角形不变量在 platesolve 中的应用综述）
//
// 约束: C++17，单线程；中文注释，UTF-8 编码
// ============================================================================

#include <vector>
#include <array>

namespace vm4 {

// --- 三角形双特征 ---
struct TriangleFeatures {
    double area;      // 面积(角秒²)
    double moment;    // 极惯性矩 J = A·(a²+b²+c²)/36
    double a, b, c;   // 三边长(角秒)，a≤b≤c
};

// 计算三角形的面积和极惯性矩
// 输入: 3 颗星的 (x,y) 坐标(角秒)
// 算法:
//   边长 a,b,c = 两两欧氏距离，排序使 a≤b≤c
//   面积 A = sqrt(s(s-a)(s-b)(s-c)), s=(a+b+c)/2 (海伦公式)
//   极惯性矩 J = A·(a²+b²+c²)/36
// 退化处理: 共线时 A 接近 0，返回全零特征
TriangleFeatures compute_triangle_features(
    double x1, double y1,
    double x2, double y2,
    double x3, double y3);

// --- 三角形验证结果 ---
struct TriangleVerifyResult {
    int    total_triangles;  // 有效三角形总数（剔除退化）
    int    passed;           // 通过验证的三角形数
    double pass_ratio;       // 通过率 = passed / total_triangles
    bool   accepted;         // 通过率 > threshold 则 true
};

// 对所有 C(n,3) 三角形组合验证几何一致性
// 输入:
//   matched_pairs: 每个元素 = {img_x, img_y, cat_x, cat_y}（角秒）
//   eps_A    : 面积相对误差阈值(默认0.05)
//   eps_J    : 极惯性矩相对误差阈值(默认0.10)
//   threshold: 通过率阈值(默认0.8)
// 算法:
//   n ≤ 30: 遍历所有 C(n,3) 组合
//   n > 30: 随机采样 min(C(n,3), 1000) 个组合（避免组合爆炸）
//   通过条件: |A_img-A_cat|/max(A_img,A_cat) < eps_A
//             |J_img-J_cat|/max(J_img,J_cat) < eps_J
//   退化三角形(A<1e-6)不计入总数
TriangleVerifyResult verify_triangles(
    const std::vector<std::array<double, 4>>& matched_pairs,
    double eps_A, double eps_J, double threshold);

} // namespace vm4

#endif // VM4_TRIANGLE_H
