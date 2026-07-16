// ============================================================================
// vm4_log.cpp - V4.0 诊断日志系统实现（Task 8）
//
// 功能:
//   - 多级日志: INFO / WARN / ERROR / DEBUG
//   - 时间戳: [YYYY-MM-DD HH:MM:SS.mmm][LEVEL][thread_id] msg
//   - 线程安全: std::mutex 保护文件写入（OpenMP 多线程并发安全）
//   - UTF-8 编码: 文件以二进制方式写入，避免 locale 影响
//   - 未 init 时退化为 stderr 输出（便于早期调试）
//
// 日志文件路径: 由 init() 指定，Task 7 在 vm4_1_solve 入口处生成
//   v4_YYYYMMDD_HHMMSS.log 并传入
//
// C++17, 单文件实现，无外部依赖
// ============================================================================

#include "vm4_log.h"

#include <chrono>
#include <cstdio>
#include <ctime>
#include <fstream>
#include <iomanip>
#include <mutex>
#include <sstream>
#include <thread>

namespace vm4_1_log {

// ----------------------------------------------------------------------------
// 全局状态（ POD 包装以便静态初始化无构造顺序问题）
// ----------------------------------------------------------------------------
static std::ofstream  g_log_file;
static std::mutex     g_log_mutex;
static bool           g_initialized = false;
static std::string    g_log_path;

// ----------------------------------------------------------------------------
// 内部工具: 生成当前时间字符串
//   格式: YYYY-MM-DD HH:MM:SS.mmm
// ----------------------------------------------------------------------------
static std::string current_timestamp() {
    using namespace std::chrono;
    auto now = system_clock::now();
    auto t_c = system_clock::to_time_t(now);
    auto ms = duration_cast<milliseconds>(now.time_since_epoch()) % 1000;

    std::tm tm_buf;
#if defined(_WIN32)
    localtime_s(&tm_buf, &t_c);
#else
    localtime_r(&t_c, &tm_buf);
#endif

    std::ostringstream oss;
    oss << std::put_time(&tm_buf, "%Y-%m-%d %H:%M:%S")
        << '.' << std::setfill('0') << std::setw(3) << ms.count();
    return oss.str();
}

// ----------------------------------------------------------------------------
// 内部工具: 获取当前线程 ID（简短表示）
// ----------------------------------------------------------------------------
static std::string thread_id_str() {
    std::ostringstream oss;
    oss << std::this_thread::get_id();
    return oss.str();
}

// ----------------------------------------------------------------------------
// 内部工具: 写入一行日志（已持有锁）
//   level_tag: "INFO" / "WARN" / "ERROR" / "DEBUG"
// ----------------------------------------------------------------------------
static void write_line_locked(const char* level_tag, const std::string& msg) {
    std::ostringstream oss;
    oss << '[' << current_timestamp() << "]"
        << '[' << level_tag << ']'
        << "[t:" << thread_id_str() << "] "
        << msg << '\n';

    const std::string line = oss.str();

    if (g_initialized && g_log_file.is_open()) {
        // 二进制写入，避免 locale 转换（保证 UTF-8 不被改写）
        g_log_file.write(line.data(), static_cast<std::streamsize>(line.size()));
        g_log_file.flush();  // 立即刷新，防止崩溃丢失日志
    } else {
        // 未 init: 退化为 stderr
        std::fputs(line.c_str(), stderr);
    }
}

// ----------------------------------------------------------------------------
// 内部工具: 通用写入入口（加锁）
// ----------------------------------------------------------------------------
static void log_impl(const char* level_tag, const std::string& msg) {
    std::lock_guard<std::mutex> lock(g_log_mutex);
    write_line_locked(level_tag, msg);
}

// ============================================================================
// init - 初始化日志文件
//   path: 日志文件完整路径（UTF-8 编码字符串）
//   若已 init 过，会先关闭旧文件再打开新文件
// ============================================================================
void init(const std::string& path) {
    std::lock_guard<std::mutex> lock(g_log_mutex);

    // 若已打开，先关闭
    if (g_log_file.is_open()) {
        g_log_file.close();
    }

    g_log_path = path;

    // 以二进制模式打开，避免 Windows 平台 \n 被自动转为 \r\n
    // 显式指定 UTF-8 字节流写入
    g_log_file.open(path, std::ios::out | std::ios::binary | std::ios::trunc);
    if (g_log_file.is_open()) {
        g_initialized = true;
        // 写入 UTF-8 BOM 头（便于 Windows 记事本识别编码）
        static const char kBOM[] = {static_cast<char>(0xEF), static_cast<char>(0xBB), static_cast<char>(0xBF)};
        g_log_file.write(kBOM, 3);
        // 写入文件头
        std::string header = "=== V4.0 Plate Solve Log ===\n";
        g_log_file.write(header.data(), static_cast<std::streamsize>(header.size()));
        g_log_file.flush();
    } else {
        g_initialized = false;
        // 退化为 stderr 输出错误
        std::fprintf(stderr, "[vm4_1_log] ERROR: 无法打开日志文件: %s\n", path.c_str());
    }
}

// ============================================================================
// 多级日志接口
// ============================================================================
void info(const std::string& msg)  { log_impl("INFO",  msg); }
void warn(const std::string& msg)  { log_impl("WARN",  msg); }
void error(const std::string& msg) { log_impl("ERROR", msg); }
void debug(const std::string& msg) { log_impl("DEBUG", msg); }

// ============================================================================
// close - 关闭日志文件
// ============================================================================
void close() {
    std::lock_guard<std::mutex> lock(g_log_mutex);
    if (g_log_file.is_open()) {
        std::string footer = "=== Log End ===\n";
        g_log_file.write(footer.data(), static_cast<std::streamsize>(footer.size()));
        g_log_file.close();
    }
    g_initialized = false;
}

} // namespace vm4_1_log
