"""
日志配置模块 (Task 1)
功能: 配置统一的日志系统，输出到文件(logs/blind_index_v2.log)和控制台，UTF-8编码
用途: 为ADV-PA盲解析所有子模块提供日志服务，记录各阶段耗时与关键指标
依赖: 标准库logging
"""
from __future__ import annotations

import logging
import sys
from pathlib import Path

# 模块根目录 (blind_index_v2/)
_MODULE_ROOT = Path(__file__).resolve().parent.parent
# 日志目录
LOG_DIR = _MODULE_ROOT / "logs"
LOG_FILE = LOG_DIR / "blind_index_v2.log"

# 全局标记，避免重复配置
_CONFIGURED = False


def setup_logging(level: int = logging.INFO, log_to_file: bool = True, log_to_console: bool = True) -> logging.Logger:
    """
    配置ADV-PA盲解析日志系统。

    Args:
        level: 日志级别，默认INFO
        log_to_file: 是否写入日志文件 (UTF-8)
        log_to_console: 是否输出到控制台

    Returns:
        根logger对象 "blind_index_v2"
    """
    global _CONFIGURED
    root_logger = logging.getLogger("blind_index_v2")
    if _CONFIGURED:
        return root_logger

    root_logger.setLevel(level)
    root_logger.propagate = False  # 不向root logger传播，避免重复输出

    # 日志格式
    fmt = logging.Formatter(
        fmt="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    if log_to_file:
        try:
            LOG_DIR.mkdir(parents=True, exist_ok=True)
            file_handler = logging.FileHandler(str(LOG_FILE), mode="a", encoding="utf-8")
            file_handler.setLevel(level)
            file_handler.setFormatter(fmt)
            root_logger.addHandler(file_handler)
        except OSError:
            pass  # 日志目录不可写时降级为仅控制台

    if log_to_console:
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(level)
        console_handler.setFormatter(fmt)
        root_logger.addHandler(console_handler)

    _CONFIGURED = True
    root_logger.info("日志系统初始化完成, 日志文件: %s", LOG_FILE)
    return root_logger


def get_logger(name: str) -> logging.Logger:
    """
    获取子模块logger。

    Args:
        name: 通常传 __name__

    Returns:
        logging.Logger 子logger
    """
    # 确保子logger挂在 "blind_index_v2" 根下
    if name.startswith("blind_index_v2"):
        logger_name = name
    else:
        logger_name = f"blind_index_v2.{name}"
    return logging.getLogger(logger_name)
