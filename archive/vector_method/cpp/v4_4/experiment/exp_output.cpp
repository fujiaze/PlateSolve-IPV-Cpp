// ============================================================================
// exp_output.cpp - V4.4 向量法抽样独立验证实验 - CSV 输出
// ============================================================================

#include "exp_output.h"
#include <fstream>
#include <iomanip>
#include <sstream>
#include <sys/stat.h>
#include <sys/types.h>

#ifdef _WIN32
#include <direct.h>
#include <errno.h>
#else
#include <errno.h>
#endif

namespace exp44 {

// ============================================================================
// 创建目录 (Windows)
// ============================================================================
static bool makeDir(const std::string& path) {
#ifdef _WIN32
    return _mkdir(path.c_str()) == 0 || errno == EEXIST;
#else
    return mkdir(path.c_str(), 0755) == 0 || errno == EEXIST;
#endif
}

// ============================================================================
// 导出 passed_pairs.csv
// ============================================================================
static int exportPassedPairs(const std::string& dir, const ExpResult& result) {
    std::string path = dir + "/passed_pairs.csv";
    std::ofstream ofs(path, std::ios::binary);
    if (!ofs.is_open()) return -1;

    // UTF-8 BOM
    const char bom[] = {(char)0xEF, (char)0xBB, (char)0xBF};
    ofs.write(bom, 3);

    ofs << "img_i,img_j,gaia_a,gaia_b,theta_deg,s_est,tx,ty,n_k_passed,similarity,vote\n";
    ofs << std::setprecision(6);
    for (const auto& p : result.passed_pairs) {
        ofs << p.img_i << ","
            << p.img_j << ","
            << p.gaia_a << ","
            << p.gaia_b << ","
            << p.theta_rot_deg << ","
            << p.s_est << ","
            << p.tx << ","
            << p.ty << ","
            << p.n_k_passed << ","
            << p.similarity << ","
            << p.vote << "\n";
    }
    return 0;
}

// ============================================================================
// 导出 focus_history.csv
// ============================================================================
static int exportFocusHistory(const std::string& dir, const ExpResult& result) {
    std::string path = dir + "/focus_history.csv";
    std::ofstream ofs(path, std::ios::binary);
    if (!ofs.is_open()) return -1;

    const char bom[] = {(char)0xEF, (char)0xBB, (char)0xBF};
    ofs.write(bom, 3);

    ofs << "sample_idx,total_votes,n_nonzero_bins,peak_cluster,snr,"
        << "peak_theta,peak_tx,peak_ty,confirmed,n_focused,n_discarded,"
        << "focus_th_lo,focus_th_hi,focus_tx_lo,focus_tx_hi,focus_ty_lo,focus_ty_hi\n";
    ofs << std::setprecision(6);
    for (const auto& s : result.focus_history) {
        ofs << s.sample_idx << ","
            << s.total_votes << ","
            << s.n_nonzero_bins << ","
            << s.peak_cluster << ","
            << s.snr << ","
            << s.peak_theta << ","
            << s.peak_tx << ","
            << s.peak_ty << ","
            << (s.confirmed ? 1 : 0) << ","
            << s.n_focused << ","
            << s.n_discarded << ","
            << s.focus_th_lo << ","
            << s.focus_th_hi << ","
            << s.focus_tx_lo << ","
            << s.focus_tx_hi << ","
            << s.focus_ty_lo << ","
            << s.focus_ty_hi << "\n";
    }
    return 0;
}

// ============================================================================
// 导出 3D 密度场切片
// ============================================================================
static int exportDensitySlice(const std::string& dir, const ExpResult& result) {
    const DensitySlice& sl = result.density_final;

    // θ-tx 投影
    {
        std::string path = dir + "/density_theta_tx.csv";
        std::ofstream ofs(path, std::ios::binary);
        if (!ofs.is_open()) return -1;
        const char bom[] = {(char)0xEF, (char)0xBB, (char)0xBF};
        ofs.write(bom, 3);

        ofs << "theta_bin,theta_deg,tx_bin,tx,votes\n";
        ofs << std::setprecision(6);
        double th_bw = (sl.th_hi - sl.th_lo) / sl.th_bins;
        double tx_bw = (sl.tx_hi - sl.tx_lo) / sl.dxdy_bins;
        for (int t = 0; t < sl.th_bins; ++t) {
            double th_val = sl.th_lo + (t + 0.5) * th_bw;
            for (int x = 0; x < sl.dxdy_bins; ++x) {
                double tx_val = sl.tx_lo + (x + 0.5) * tx_bw;
                double v = sl.theta_tx[(size_t)t * sl.dxdy_bins + x];
                if (v > 0) {
                    ofs << t << "," << th_val << "," << x << "," << tx_val << "," << v << "\n";
                }
            }
        }
    }

    // θ-ty 投影
    {
        std::string path = dir + "/density_theta_ty.csv";
        std::ofstream ofs(path, std::ios::binary);
        if (!ofs.is_open()) return -1;
        const char bom[] = {(char)0xEF, (char)0xBB, (char)0xBF};
        ofs.write(bom, 3);

        ofs << "theta_bin,theta_deg,ty_bin,ty,votes\n";
        ofs << std::setprecision(6);
        double th_bw = (sl.th_hi - sl.th_lo) / sl.th_bins;
        double ty_bw = (sl.ty_hi - sl.ty_lo) / sl.dxdy_bins;
        for (int t = 0; t < sl.th_bins; ++t) {
            double th_val = sl.th_lo + (t + 0.5) * th_bw;
            for (int y = 0; y < sl.dxdy_bins; ++y) {
                double ty_val = sl.ty_lo + (y + 0.5) * ty_bw;
                double v = sl.theta_ty[(size_t)t * sl.dxdy_bins + y];
                if (v > 0) {
                    ofs << t << "," << th_val << "," << y << "," << ty_val << "," << v << "\n";
                }
            }
        }
    }

    // tx-ty 投影
    {
        std::string path = dir + "/density_tx_ty.csv";
        std::ofstream ofs(path, std::ios::binary);
        if (!ofs.is_open()) return -1;
        const char bom[] = {(char)0xEF, (char)0xBB, (char)0xBF};
        ofs.write(bom, 3);

        ofs << "tx_bin,tx,ty_bin,ty,votes\n";
        ofs << std::setprecision(6);
        double tx_bw = (sl.tx_hi - sl.tx_lo) / sl.dxdy_bins;
        double ty_bw = (sl.ty_hi - sl.ty_lo) / sl.dxdy_bins;
        for (int x = 0; x < sl.dxdy_bins; ++x) {
            double tx_val = sl.tx_lo + (x + 0.5) * tx_bw;
            for (int y = 0; y < sl.dxdy_bins; ++y) {
                double ty_val = sl.ty_lo + (y + 0.5) * ty_bw;
                double v = sl.tx_ty[(size_t)x * sl.dxdy_bins + y];
                if (v > 0) {
                    ofs << x << "," << tx_val << "," << y << "," << ty_val << "," << v << "\n";
                }
            }
        }
    }

    return 0;
}

// ============================================================================
// 导出 inlier_pairs.csv (RANSAC inliers 点对关系)
// ============================================================================
static int exportInlierPairs(const std::string& dir, const ExpResult& result) {
    std::string path = dir + "/inlier_pairs.csv";
    std::ofstream ofs(path, std::ios::binary);
    if (!ofs.is_open()) return -1;

    const char bom[] = {(char)0xEF, (char)0xBB, (char)0xBF};
    ofs.write(bom, 3);

    ofs << "img_idx,gaia_idx,s_est,theta_rot_deg\n";
    ofs << std::setprecision(6);
    for (const auto& p : result.inlier_pairs) {
        ofs << p.img_idx << ","
            << p.gaia_idx << ","
            << p.s_est << ","
            << p.theta_rot << "\n";
    }
    return 0;
}

// ============================================================================
// 导出 result_summary.csv
// ============================================================================
static int exportResultSummary(const std::string& dir, const ExpResult& result, const ExpInput& input) {
    std::string path = dir + "/result_summary.csv";
    std::ofstream ofs(path, std::ios::binary);
    if (!ofs.is_open()) return -1;

    const char bom[] = {(char)0xEF, (char)0xBB, (char)0xBF};
    ofs.write(bom, 3);

    ofs << "field,value\n";
    ofs << std::setprecision(10);

    // 基本信息
    ofs << "data_name," << input.data_name << "\n";
    ofs << "source," << (input.source == DataSource::SYNTHETIC ? "synthetic" : "real") << "\n";
    ofs << "n_u," << input.U.size() << "\n";
    ofs << "n_w," << input.W.size() << "\n";
    ofs << "s0," << input.s0 << "\n";

    // 密度场峰值 (粗略)
    ofs << "density_theta_deg," << (result.density_estimate.theta * 180.0 / 3.14159265358979323846) << "\n";
    ofs << "density_tx," << result.density_estimate.tx << "\n";
    ofs << "density_ty," << result.density_estimate.ty << "\n";
    ofs << "snr_final," << result.snr_final << "\n";
    ofs << "n_samples_actual," << result.n_samples_actual << "\n";
    ofs << "n_passed," << result.n_passed << "\n";
    ofs << "n_focused," << result.n_focused << "\n";

    // ⭐ RANSAC 一步求解结果 (最终输出)
    ofs << "ransac_theta_deg," << (result.ransac_estimate.theta * 180.0 / 3.14159265358979323846) << "\n";
    ofs << "ransac_tx," << result.ransac_estimate.tx << "\n";
    ofs << "ransac_ty," << result.ransac_estimate.ty << "\n";
    ofs << "ransac_s," << result.ransac_estimate.s << "\n";
    ofs << "n_inliers," << result.n_inliers << "\n";
    ofs << "n_ransac_iters," << result.n_ransac_iters << "\n";
    ofs << "ransac_rms," << result.ransac_rms << "\n";
    ofs << "success," << (result.success ? 1 : 0) << "\n";

    // 真值 (模拟数据)
    if (input.has_ground_truth && input.ground_truth.valid) {
        ofs << "gt_theta_deg," << (input.ground_truth.theta * 180.0 / 3.14159265358979323846) << "\n";
        ofs << "gt_tx," << input.ground_truth.tx << "\n";
        ofs << "gt_ty," << input.ground_truth.ty << "\n";
        ofs << "gt_s," << input.ground_truth.s << "\n";
        ofs << "err_theta_deg," << result.err_theta_deg << "\n";
        ofs << "err_tx," << result.err_tx << "\n";
        ofs << "err_ty," << result.err_ty << "\n";
        ofs << "err_s," << result.err_s << "\n";
    }

    return 0;
}

// ============================================================================
// 导出全部 CSV
// ============================================================================
int exportAllCSV(
    const std::string& output_dir,
    const ExpResult& result,
    const ExpInput& input)
{
    makeDir(output_dir);

    exportPassedPairs(output_dir, result);
    exportInlierPairs(output_dir, result);
    exportFocusHistory(output_dir, result);
    exportDensitySlice(output_dir, result);
    exportResultSummary(output_dir, result, input);

    return 0;
}

} // namespace exp44
