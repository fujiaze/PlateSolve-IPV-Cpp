#ifndef WF_API_H
#define WF_API_H

// ============================================================================
// wf_api.h - V4.2 WcsFitter 模块 C 接口（Task 6）
//
// 职责:
//   Phase E 分层 SIP 拟合:
//     Layer 0: Umeyama SVD → CD 矩阵
//     Layer 1: 像素残差 MAD 剔除 outlier → 6 参数全仿射 → 更新 CD/CRVAL
//     Layer 2: BIC 选择 SIP 阶数 (2-4 阶, 高阶需 BIC 差 > 2 才选)
//
// 输入: U(N×2, 图像侧角秒坐标, Y轴向上), W(M×2, 星表侧角秒坐标, 已投影到切平面)
//       pairs_u[n_pairs], pairs_w[n_pairs]: 匹配对索引
// 输出: WcsResult (cd[4], crval[2], crpix[2], sip_A[36], sip_B[36], sip_order, rms_px, n_pairs, success)
//
// 约束: V4.2 无 flip_mode (VectorMatcher 已统一方向), CD 为标准 WCS 格式
// ============================================================================

#include "v42_types.h"

// 将 v42::WcsResult 引入全局命名空间, 简化 C 接口声明
using v42::WcsResult;

#ifdef _WIN32
#define WF_API __declspec(dllexport)
#else
#define WF_API __attribute__((visibility("default")))
#endif

// WCS 拟合参数
struct WcsFitterParams {
    double s0;              // 像素尺度(角秒/像素)
    int    sip_max_order;   // SIP最大阶数(默认4)
    int    skip_sip;        // 跳过SIP拟合, 仅线性CD(默认0)
    double img_width;       // 图像宽度(像素)
    double img_height;      // 图像高度(像素)
    double center_ra;       // 中心赤经(度) - CRVAL1
    double center_dec;      // 中心赤纬(度) - CRVAL2
    const char* log_file_path;  // 日志文件路径(UTF-8, NULL=仅stderr)
};

#ifdef __cplusplus
extern "C" {
#endif

// 执行 WCS 分层拟合
// U[N_img×2]: 图像星角秒坐标(原点在图像中心, Y轴向上)
// W[M×2]: 星表星角秒坐标(已投影到切平面)
// pairs_u[n_pairs], pairs_w[n_pairs]: 匹配对索引
// 返回: 0=失败, 1=成功
WF_API int wf_fit(
    const double* U, int N_img,
    const double* W, int M,
    const int* pairs_u, const int* pairs_w, int n_pairs,
    const WcsFitterParams* params,
    WcsResult* result
);

#ifdef __cplusplus
}
#endif

#endif // WF_API_H
