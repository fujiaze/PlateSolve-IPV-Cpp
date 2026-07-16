#ifndef V42_TYPES_H
#define V42_TYPES_H

#include <vector>
#include <string>
#include <cstdint>

namespace v42 {

// 相似变换参数 (s, θ, tx, ty)
struct SimTransform {
    double s;       // 尺度
    double theta;   // 旋转角(弧度)
    double tx;      // X平移
    double ty;      // Y平移
    bool   valid;   // 是否有效
};

// 匹配对 (图像星索引 u, 星表星索引 w)
struct MatchPair {
    int u;
    int w;
};

// 星点 (角秒坐标, 原点在图像中心)
struct StarPoint {
    double x;       // X坐标(角秒)
    double y;       // Y坐标(角秒)
    double flux;    // 流量
    bool   saturated; // 是否饱和
};

// 区域统计
struct RegionStats {
    int region_idx;
    int n_pairs;        // 该区域匹配对数
    int n_total_stars;  // 该区域总星数
    bool is_sparse;     // 是否为稀疏区
};

// 贝叶斯验证结果
struct BayesResult {
    double lnK;          // 贝叶斯因子对数
    int    n_match;      // 匹配对数
    double rms_arcsec;   // RMS(角秒)
    double sigma;        // 位置噪声σ
    int    decision;     // 1=接受, 0=弱证据, -1=拒绝
};

// 三角形验证结果
struct TriangleResult {
    int    total;        // 三角形总数
    int    passed;       // 通过数
    double pass_ratio;   // 通过率
    bool   accepted;     // 是否通过
};

// WCS 结果
struct WcsResult {
    double cd[4];        // CD矩阵 [cd11, cd12, cd21, cd22]
    double crval[2];     // 中心赤经赤纬(度)
    double crpix[2];     // 参考像素(1-based)
    double sip_A[36];    // SIP A 多项式 (最多4阶, 6x6=36)
    double sip_B[36];    // SIP B 多项式
    int    sip_order;    // 实际SIP阶数
    double rms_px;       // RMS(像素)
    int    n_pairs;      // 拟合用对数
    bool   success;      // 是否成功
};

} // namespace v42

#endif // V42_TYPES_H
