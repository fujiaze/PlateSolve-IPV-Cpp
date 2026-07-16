// ============================================================================
// exp_main.cpp - V4.4 向量法抽样独立验证实验 - 主程序
//
// 用法:
//   exp_experiment.exe --synthetic [--output_dir DIR] [--seed N]
//   exp_experiment.exe --real --image PATH --ra RA --dec DEC
//                      --focal F --pixel P [--output_dir DIR]
//
// 模拟数据模式 (--synthetic):
//   生成模拟 U/W (已知真值), 运行向量法, 输出 CSV
//
// 真实数据模式 (--real):
//   调用 vm44_select 获取真实 U/W, 运行向量法, 输出 CSV
//   注意: 需要先注入 GaiaClient 和 StarDetector 句柄
//         (通过环境变量或后续 Python 脚本注入)
// ============================================================================

#include "exp_types.h"
#include "exp_relvec_core.h"
#include "exp_data_source.h"
#include "exp_output.h"
#include "vm44_api.h"

#include <cstdio>
#include <cstring>
#include <string>
#include <vector>
#include <iostream>

#ifdef _WIN32
#include <windows.h>
#endif

// ============================================================================
// 命令行参数解析
// ============================================================================
struct CmdArgs {
    bool   synthetic = false;
    bool   real = false;
    std::string output_dir = "./exp_output";
    std::string image_path;
    double ra = 0, dec = 0;
    double focal_length_mm = 0;
    double pixel_size_um = 0;
    int    seed = 42;
    int    n_stars = 100;
    double theta_true_deg = 30.0;
    bool   help = false;
};

static bool parseArgs(int argc, char** argv, CmdArgs& args) {
    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        if (arg == "--help" || arg == "-h") {
            args.help = true;
        } else if (arg == "--synthetic") {
            args.synthetic = true;
        } else if (arg == "--real") {
            args.real = true;
        } else if (arg == "--output_dir" && i + 1 < argc) {
            args.output_dir = argv[++i];
        } else if (arg == "--image" && i + 1 < argc) {
            args.image_path = argv[++i];
        } else if (arg == "--ra" && i + 1 < argc) {
            args.ra = std::stod(argv[++i]);
        } else if (arg == "--dec" && i + 1 < argc) {
            args.dec = std::stod(argv[++i]);
        } else if (arg == "--focal" && i + 1 < argc) {
            args.focal_length_mm = std::stod(argv[++i]);
        } else if (arg == "--pixel" && i + 1 < argc) {
            args.pixel_size_um = std::stod(argv[++i]);
        } else if (arg == "--seed" && i + 1 < argc) {
            args.seed = std::stoi(argv[++i]);
        } else if (arg == "--n_stars" && i + 1 < argc) {
            args.n_stars = std::stoi(argv[++i]);
        } else if (arg == "--theta_true" && i + 1 < argc) {
            args.theta_true_deg = std::stod(argv[++i]);
        } else {
            std::fprintf(stderr, "未知参数: %s\n", arg.c_str());
            return false;
        }
    }
    return true;
}

static void printHelp() {
    std::fprintf(stderr,
        "用法:\n"
        "  模拟数据: exp_experiment.exe --synthetic [--output_dir DIR] [--seed N] [--n_stars N] [--theta_true DEG]\n"
        "  真实数据: exp_experiment.exe --real --image PATH --ra RA --dec DEC --focal F --pixel P [--output_dir DIR]\n"
        "\n"
        "选项:\n"
        "  --synthetic       模拟数据模式\n"
        "  --real            真实数据模式\n"
        "  --output_dir DIR  输出目录 (默认 ./exp_output)\n"
        "  --image PATH      FITS 图像路径\n"
        "  --ra RA           中心赤经 (度)\n"
        "  --dec DEC         中心赤纬 (度)\n"
        "  --focal F         焦距 (mm)\n"
        "  --pixel P         像元尺寸 (um)\n"
        "  --seed N          随机种子 (默认 42)\n"
        "  --n_stars N       模拟星点数 (默认 100)\n"
        "  --theta_true DEG  模拟真值 θ (度, 默认 30)\n"
    );
}

// ============================================================================
// main
// ============================================================================
int main(int argc, char** argv) {
    // UTF-8 输出
#ifdef _WIN32
    SetConsoleOutputCP(CP_UTF8);
#endif

    CmdArgs args;
    if (!parseArgs(argc, argv, args) || args.help) {
        printHelp();
        return args.help ? 0 : 1;
    }

    if (!args.synthetic && !args.real) {
        std::fprintf(stderr, "错误: 必须指定 --synthetic 或 --real\n");
        printHelp();
        return 1;
    }

    exp44::ExpInput input;

    if (args.synthetic) {
        // 模拟数据模式
        exp44::SyntheticParams sp = exp44::getDefaultSyntheticParams();
        sp.seed = args.seed;
        sp.n_stars = args.n_stars;
        sp.theta_true_deg = args.theta_true_deg;

        if (exp44::generateSyntheticData(sp, input, "synthetic") != 0) {
            std::fprintf(stderr, "生成模拟数据失败\n");
            return 1;
        }
        std::fprintf(stderr, "=== 模拟数据 ===\n");
        std::fprintf(stderr, "N_u=%zu N_w=%zu s0=%.4f\n",
                     input.U.size(), input.W.size(), input.s0);
        std::fprintf(stderr, "真值: θ=%.2f° tx=%.2f ty=%.2f s=%.4f\n",
                     sp.theta_true_deg, sp.tx_true, sp.ty_true, sp.s_true);
    } else {
        // 真实数据模式
        if (args.image_path.empty() || args.focal_length_mm <= 0 || args.pixel_size_um <= 0) {
            std::fprintf(stderr, "错误: --real 模式需要 --image --focal --pixel (可选 --ra --dec)\n");
            return 1;
        }

        exp44::RealDataParams rp;
        rp.image_path = args.image_path;
        rp.ra = args.ra;
        rp.dec = args.dec;
        rp.focal_length_mm = args.focal_length_mm;
        rp.pixel_size_um = args.pixel_size_um;

        if (exp44::loadRealData(rp, input, args.image_path) != 0) {
            std::fprintf(stderr, "加载真实数据失败\n");
            return 1;
        }
        std::fprintf(stderr, "=== 真实数据 ===\n");
        std::fprintf(stderr, "N_u=%zu N_w=%zu s0=%.4f\n",
                     input.U.size(), input.W.size(), input.s0);
    }

    // 运行实验
    exp44::RelVecParams params = exp44::getDefaultRelVecParams();
    params.seed = args.seed;

    exp44::ExpResult result;
    std::string log_dir = args.output_dir + "/logs";
    int rc = exp44::runRelVecExperiment(
        input.U, input.W, input.s0,
        params,
        input.ground_truth, input.has_ground_truth,
        result,
        log_dir
    );

    if (rc != 0) {
        std::fprintf(stderr, "实验失败\n");
        return 1;
    }

    // 导出 CSV
    if (exp44::exportAllCSV(args.output_dir, result, input) != 0) {
        std::fprintf(stderr, "导出 CSV 失败\n");
        return 1;
    }

    std::fprintf(stderr, "\n=== 实验完成 ===\n");
    std::fprintf(stderr, "输出目录: %s\n", args.output_dir.c_str());
    std::fprintf(stderr, "CSV 文件:\n");
    std::fprintf(stderr, "  passed_pairs.csv     - 通过候选\n");
    std::fprintf(stderr, "  focus_history.csv    - 递归聚焦过程\n");
    std::fprintf(stderr, "  density_theta_tx.csv - θ-tx 密度场投影\n");
    std::fprintf(stderr, "  density_theta_ty.csv - θ-ty 密度场投影\n");
    std::fprintf(stderr, "  density_tx_ty.csv    - tx-ty 密度场投影\n");
    std::fprintf(stderr, "  result_summary.csv   - 结果汇总\n");

    return 0;
}
