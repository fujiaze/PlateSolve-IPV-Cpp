"""
FITS 头指向读取辅助 (Bug 1 修复)
功能: 从 FITS 头读取 RA/DEC 指向, 支持 WCS CRVAL1/2 和 OBJCTRA/OBJCTDEC/RA/DEC 两种来源
用途: DD-SPPS 测试 harness 构建本地 Gaia 模板时获取指向中心 (当 WCS 不存在时的回退方案)
依赖: lib.astro_image_io.python.astro_image_io (ImageReader)
"""
from __future__ import annotations

from typing import Optional, Tuple

from lib.astro_image_io.python.astro_image_io import ImageReader
from lib.plate_solve.blind_index_v2.python.logging_setup import get_logger

logger = get_logger("ddspps.io_helpers")


def _strip_quotes(val: str) -> str:
    """去除 FITS 头字符串值两端的引号和空白。"""
    s = val.strip()
    # 去除单引号或双引号 (FITS 标准用单引号, 但兼容双引号)
    if len(s) >= 2 and s[0] == s[-1] and s[0] in ("'", '"'):
        s = s[1:-1].strip()
    return s


def _parse_ra(ra_str: str) -> Optional[float]:
    """
    解析 RA 字符串 → 度。

    支持:
        - "HH MM SS.SS" (空格分隔) → (HH + MM/60 + SS/3600) × 15 度
        - "HH:MM:SS.SS" (冒号分隔) → 同上
        - "HH:MM:SS.SS" 或 "HH MM SS.SS" (含可选符号)
        - 纯数字 (已是度) → 直接 float()

    Args:
        ra_str: RA 字符串

    Returns:
        RA (度), 解析失败返回 None
    """
    if ra_str is None:
        return None
    s = _strip_quotes(str(ra_str))
    if not s:
        return None

    # 尝试纯数字 (已是度)
    try:
        return float(s)
    except ValueError:
        pass

    # 替换冒号为空格, 统一处理
    s_norm = s.replace(":", " ").strip()
    parts = s_norm.split()
    if len(parts) < 2 or len(parts) > 3:
        logger.warning("RA 解析失败 (无法识别格式): %r", ra_str)
        return None

    try:
        hh = float(parts[0])
        mm = float(parts[1])
        ss = float(parts[2]) if len(parts) == 3 else 0.0
    except ValueError:
        logger.warning("RA 解析失败 (非数值): %r", ra_str)
        return None

    if mm < 0 or ss < 0:
        logger.warning("RA 解析失败 (分/秒为负): %r", ra_str)
        return None

    ra_deg = (hh + mm / 60.0 + ss / 3600.0) * 15.0
    # 归一化到 [0, 360)
    ra_deg = ra_deg % 360.0
    return float(ra_deg)


def _parse_dec(dec_str: str) -> Optional[float]:
    """
    解析 DEC 字符串 → 度。

    支持:
        - "DD MM SS.SS" (空格分隔) → ±(DD + MM/60 + SS/3600) 度
        - "DD:MM:SS.SS" (冒号分隔) → 同上
        - 负号处理: "-22 51 00" → -(22 + 51/60 + 0/3600)
        - 纯数字 (已是度) → 直接 float()

    Args:
        dec_str: DEC 字符串

    Returns:
        Dec (度), 解析失败返回 None
    """
    if dec_str is None:
        return None
    s = _strip_quotes(str(dec_str))
    if not s:
        return None

    # 尝试纯数字 (已是度)
    try:
        return float(s)
    except ValueError:
        pass

    # 提取符号
    sign = 1.0
    if s.startswith("-"):
        sign = -1.0
        s = s[1:].strip()
    elif s.startswith("+"):
        s = s[1:].strip()

    # 替换冒号为空格, 统一处理
    s_norm = s.replace(":", " ").strip()
    parts = s_norm.split()
    if len(parts) < 2 or len(parts) > 3:
        logger.warning("DEC 解析失败 (无法识别格式): %r", dec_str)
        return None

    try:
        dd = float(parts[0])
        mm = float(parts[1])
        ss = float(parts[2]) if len(parts) == 3 else 0.0
    except ValueError:
        logger.warning("DEC 解析失败 (非数值): %r", dec_str)
        return None

    if mm < 0 or ss < 0:
        logger.warning("DEC 解析失败 (分/秒为负): %r", dec_str)
        return None

    dec_deg = sign * (dd + mm / 60.0 + ss / 3600.0)
    return float(dec_deg)


def get_pointing_from_fits(path: str) -> Optional[Tuple[float, float]]:
    """
    从 FITS 头读取指向 (RA, Dec) 度。

    优先级:
        1. CRVAL1/CRVAL2 (WCS, 已是度)
        2. OBJCTRA/OBJCTDEC (字符串格式 "HH MM SS.SS")
        3. RA/DEC (字符串格式 "HH MM SS.SS")

    Args:
        path: FITS/XISF 文件路径

    Returns:
        (ra_deg, dec_deg), 失败返回 None
    """
    reader = ImageReader()
    img_data = None
    try:
        img_data = reader.read_header_only(path)
    except Exception as e:
        logger.error("读取 FITS 头失败: %s, 错误: %s", path, e)
        return None

    try:
        # 1. 尝试 CRVAL1/CRVAL2 (WCS, 已是度)
        crval1 = img_data.get_keyword_float("CRVAL1", default=float("nan"))
        crval2 = img_data.get_keyword_float("CRVAL2", default=float("nan"))
        # NaN 检查 (crval1 != crval1 即 NaN)
        if not (crval1 != crval1) and not (crval2 != crval2):
            if crval1 != 0.0 or crval2 != 0.0:
                logger.info("从 CRVAL 读取指向: RA=%.6f, Dec=%.6f", crval1, crval2)
                return float(crval1), float(crval2)

        # 2. 尝试 OBJCTRA/OBJCTDEC 或 RA/DEC
        for ra_kw in ("OBJCTRA", "RA"):
            ra_val = img_data.get_keyword(ra_kw)
            if ra_val:
                ra = _parse_ra(ra_val)
                if ra is not None:
                    for dec_kw in ("OBJCTDEC", "DEC"):
                        dec_val = img_data.get_keyword(dec_kw)
                        if dec_val:
                            dec = _parse_dec(dec_val)
                            if dec is not None:
                                logger.info("从 %s/%s 读取指向: RA=%.6f, Dec=%.6f (原始: %r / %r)",
                                            ra_kw, dec_kw, ra, dec, ra_val, dec_val)
                                return ra, dec

        logger.warning("无法从 FITS 头读取指向: %s", path)
        return None
    finally:
        if img_data is not None:
            img_data.close()
