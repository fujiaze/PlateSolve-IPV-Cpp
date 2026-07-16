// ============================================================================
// exp_dll.cpp - V4.4 向量法抽样独立验证实验 - DLL 接口
//
// 暴露 C 接口供 Python 批量测试脚本调用:
//   exp_set_gaia_client(handle)   - 注入 GaiaClient 句柄
//   exp_set_star_detector(handle) - 注入 StarDetector 句柄
//   exp_solve(image, ra, dec, focal, pixel, output_dir) -> int
//   exp_solve_synthetic(output_dir, seed, n_stars, theta_true) -> int
//   exp_get_result_summary(output_dir) -> ExpResultSummary (结构体)
//
// Python 通过 ctypes 调用, 复用 V4.4 的 GaiaClientPy + StarDetector 基础设施
// ============================================================================

#include "exp_types.h"
#include "exp_relvec_core.h"
#include "exp_data_source.h"
#include "exp_output.h"
#include "vm44_api.h"

#include <cstring>
#include <string>
#include <cstdio>

#ifdef _WIN32
#define EXP_API __declspec(dllexport)
#else
#define EXP_API __attribute__((visibility("default")))
#endif

// ============================================================================
// C 接口
// ============================================================================
extern "C" {

// 注入 GaiaClient 句柄 (转发给 vm44_set_gaia_client)
EXP_API void exp_set_gaia_client(void* handle) {
    vm44_set_gaia_client(handle);
}

// 注入 StarDetector 句柄
EXP_API void exp_set_star_detector(void* handle) {
    vm44_set_star_detector(handle);
}

// 结果摘要 (供 Python 读取)
struct ExpResultSummary {
    int    success;
    double ransac_theta_deg;
    double ransac_tx;
    double ransac_ty;
    double ransac_s;
    int    n_inliers;
    int    n_ransac_iters;
    double ransac_rms;
    double snr_final;
    int    n_samples_actual;
    int    n_passed;
    int    n_focused;
    // 误差 (仅模拟数据)
    double err_theta_deg;
    double err_tx;
    double err_ty;
    double err_s;
    char   error_msg[256];
};

// 真实数据求解
EXP_API int exp_solve(
    const char* image_path,
    double ra, double dec,
    double focal_length_mm,
    double pixel_size_um,
    const char* output_dir,
    ExpResultSummary* summary)
{
    if (summary) {
        std::memset(summary, 0, sizeof(*summary));
    }

    exp44::RealDataParams rp;
    rp.image_path = image_path;
    rp.ra = ra;
    rp.dec = dec;
    rp.focal_length_mm = focal_length_mm;
    rp.pixel_size_um = pixel_size_um;

    exp44::ExpInput input;
    if (exp44::loadRealData(rp, input, image_path) != 0) {
        if (summary) {
            std::strncpy(summary->error_msg, "loadRealData 失败", sizeof(summary->error_msg) - 1);
        }
        return -1;
    }

    exp44::RelVecParams params = exp44::getDefaultRelVecParams();
    exp44::ExpResult result;
    std::string log_dir = std::string(output_dir) + "/logs";
    int rc = exp44::runRelVecExperiment(
        input.U, input.W, input.s0,
        params,
        input.ground_truth, input.has_ground_truth,
        result,
        log_dir
    );
    if (rc != 0) {
        if (summary) {
            std::strncpy(summary->error_msg, "runRelVecExperiment 失败", sizeof(summary->error_msg) - 1);
        }
        return -1;
    }

    if (exp44::exportAllCSV(output_dir, result, input) != 0) {
        if (summary) {
            std::strncpy(summary->error_msg, "exportAllCSV 失败", sizeof(summary->error_msg) - 1);
        }
        return -1;
    }

    // 填充摘要
    if (summary) {
        summary->success = result.success ? 1 : 0;
        summary->ransac_theta_deg = result.ransac_estimate.theta * 180.0 / 3.14159265358979323846;
        summary->ransac_tx = result.ransac_estimate.tx;
        summary->ransac_ty = result.ransac_estimate.ty;
        summary->ransac_s = result.ransac_estimate.s;
        summary->n_inliers = result.n_inliers;
        summary->n_ransac_iters = result.n_ransac_iters;
        summary->ransac_rms = result.ransac_rms;
        summary->snr_final = result.snr_final;
        summary->n_samples_actual = result.n_samples_actual;
        summary->n_passed = result.n_passed;
        summary->n_focused = result.n_focused;
        summary->err_theta_deg = result.err_theta_deg;
        summary->err_tx = result.err_tx;
        summary->err_ty = result.err_ty;
        summary->err_s = result.err_s;
    }

    return 0;
}

// 模拟数据求解 (供快速验证)
EXP_API int exp_solve_synthetic(
    int seed, int n_stars, double theta_true_deg,
    const char* output_dir,
    ExpResultSummary* summary)
{
    if (summary) {
        std::memset(summary, 0, sizeof(*summary));
    }

    exp44::SyntheticParams sp = exp44::getDefaultSyntheticParams();
    sp.seed = seed;
    sp.n_stars = n_stars;
    sp.theta_true_deg = theta_true_deg;

    exp44::ExpInput input;
    if (exp44::generateSyntheticData(sp, input, "synthetic") != 0) {
        if (summary) {
            std::strncpy(summary->error_msg, "generateSyntheticData 失败", sizeof(summary->error_msg) - 1);
        }
        return -1;
    }

    exp44::RelVecParams params = exp44::getDefaultRelVecParams();
    params.seed = seed;
    exp44::ExpResult result;
    std::string log_dir = std::string(output_dir) + "/logs";
    int rc = exp44::runRelVecExperiment(
        input.U, input.W, input.s0,
        params,
        input.ground_truth, input.has_ground_truth,
        result,
        log_dir
    );
    if (rc != 0) {
        if (summary) {
            std::strncpy(summary->error_msg, "runRelVecExperiment 失败", sizeof(summary->error_msg) - 1);
        }
        return -1;
    }

    if (exp44::exportAllCSV(output_dir, result, input) != 0) {
        if (summary) {
            std::strncpy(summary->error_msg, "exportAllCSV 失败", sizeof(summary->error_msg) - 1);
        }
        return -1;
    }

    if (summary) {
        summary->success = result.success ? 1 : 0;
        summary->ransac_theta_deg = result.ransac_estimate.theta * 180.0 / 3.14159265358979323846;
        summary->ransac_tx = result.ransac_estimate.tx;
        summary->ransac_ty = result.ransac_estimate.ty;
        summary->ransac_s = result.ransac_estimate.s;
        summary->n_inliers = result.n_inliers;
        summary->n_ransac_iters = result.n_ransac_iters;
        summary->ransac_rms = result.ransac_rms;
        summary->snr_final = result.snr_final;
        summary->n_samples_actual = result.n_samples_actual;
        summary->n_passed = result.n_passed;
        summary->n_focused = result.n_focused;
        summary->err_theta_deg = result.err_theta_deg;
        summary->err_tx = result.err_tx;
        summary->err_ty = result.err_ty;
        summary->err_s = result.err_s;
    }

    return 0;
}

} // extern "C"
