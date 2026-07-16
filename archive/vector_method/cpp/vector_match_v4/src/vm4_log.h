#ifndef VM4_LOG_H
#define VM4_LOG_H

// ============================================================================
// vm4_log.h - V4.0 日志系统声明（骨架）
//
// 具体实现由后续 Task 8 完善，本文件仅给出函数声明骨架。
// vm4_core.cpp 通过 #include "vm4_log.h" 引入这些声明，保证编译通过。
// ============================================================================

#include <string>

namespace vm4_log {
    void init(const std::string& path);  // 初始化日志文件
    void info(const std::string& msg);
    void warn(const std::string& msg);
    void error(const std::string& msg);
    void debug(const std::string& msg);
    void close();
}

#endif
