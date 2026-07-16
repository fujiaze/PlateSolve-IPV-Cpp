"""
统一IO接口封装层 (Task 1)
功能: 封装现有star_detector/gaia_client/astro_image_io三个DLL的Python绑定，提供统一接口
用途: 4SADQ-KV盲解析管线通过此层调用星点检测、Gaia DR3查询、图像读取，避免重复开发
依赖:
    - lib.star_detector.python.star_detector (StarDetector, SDetParamsPy)
    - lib.plate_solve.python.vector_match_v2 (GaiaClientPy, gnomonic_forward, gnomonic_inverse)
    - lib.astro_image_io.python.astro_image_io (ImageReader, ImageMetadataPy)
"""
from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np

# 复用现有DLL封装
from lib.star_detector.python.star_detector import StarDetector, SDetParamsPy
from lib.plate_solve.python.vector_match_v2 import GaiaClientPy, gnomonic_forward, gnomonic_inverse
from lib.astro_image_io.python.astro_image_io import ImageReader, ImageMetadataPy

from .logging_setup import get_logger

logger = get_logger(__name__)

# 项目根目录 (用于默认GaiaDR3数据路径)
_PROJECT_ROOT = os.path.normpath(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..", "..")))
_DEFAULT_GAIA_DR3_DIR = os.path.join(_PROJECT_ROOT, "GaiaDR3")


@dataclass
class StarDetectionResult:
    """星点检测结果容器"""
    x: np.ndarray          # shape (N,) 像素x坐标
    y: np.ndarray          # shape (N,) 像素y坐标
    flux: np.ndarray       # shape (N,) 流量(正常星=振幅A, 饱和星=-1)
    saturated: np.ndarray  # shape (N,) 0=正常星, 1=饱和星

    @property
    def count(self) -> int:
        return len(self.x)


def read_image(path: str) -> Tuple[np.ndarray, ImageMetadataPy]:
    """
    读取天文图像(FITS/XISF自动检测)，返回uint16数组与元数据。

    Args:
        path: 图像文件路径

    Returns:
        (uint16_img, metadata): uint16 ndarray (H,W), ImageMetadataPy
    """
    logger.info("读取图像: %s", path)
    reader = ImageReader()
    img_data = reader.read(path)
    try:
        raw = img_data.data  # float32
        # 转为uint16
        uint16_img = np.clip(raw, 0, 65535).astype(np.uint16)
        metadata = img_data.metadata
        logger.info("图像读取完成: %dx%d, dtype=%s -> uint16, 有WCS=%s",
                     img_data.width, img_data.height, raw.dtype,
                     metadata.wcs is not None and metadata.wcs.has_wcs)
        return uint16_img, metadata
    finally:
        img_data.close()


def detect_stars(
    uint16_img: np.ndarray,
    params: Optional[SDetParamsPy] = None,
) -> StarDetectionResult:
    """
    检测图像星点。

    Args:
        uint16_img: uint16 2D数组
        params: 检测参数，默认maxStars=0(不截断), fitRadius=6

    Returns:
        StarDetectionResult: x/y/flux/saturated数组
    """
    if params is None:
        params = SDetParamsPy(maxStars=0, fitRadius=6)
    logger.info("星点检测开始, 参数: maxStars=%d, fitRadius=%d", params.maxStars, params.fitRadius)
    detector = StarDetector(params=params)
    try:
        result = detector.detect_ex(uint16_img)
        x_arr = np.array(result.x, dtype=np.float64)
        y_arr = np.array(result.y, dtype=np.float64)
        flux_arr = np.array(result.flux, dtype=np.float64)
        saturated_arr = np.array(result.saturated, dtype=np.int32)
        logger.info("星点检测完成: %d颗星 (%d正常, %d饱和)",
                     result.count, result.normal_count, result.saturated_count)
        return StarDetectionResult(x=x_arr, y=y_arr, flux=flux_arr, saturated=saturated_arr)
    finally:
        detector.close()


def query_dr3(
    ra0: float,
    dec0: float,
    radius_deg: float,
    mag_limit: float,
    data_dir: Optional[str] = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    查询Gaia DR3局部天区参考星。

    Args:
        ra0: 中心RA(度)
        dec0: 中心Dec(度)
        radius_deg: 锥形搜索半径(度)
        mag_limit: 极限星等
        data_dir: GaiaDR3数据目录，默认 <project_root>/GaiaDR3

    Returns:
        (ra, dec, mag): 三个numpy数组
    """
    if data_dir is None:
        data_dir = _DEFAULT_GAIA_DR3_DIR
    logger.info("查询DR3: 中心(%.5f, %.5f), 半径=%.4f°, 极限星等=%.2f, 数据目录=%s",
                 ra0, dec0, radius_deg, mag_limit, data_dir)
    # db_type=1 → DR3
    client = GaiaClientPy(data_dir=data_dir, db_type=1)
    try:
        ra, dec, mag = client.cone_search(ra0, dec0, radius_deg, mag_limit)
        n = len(ra)
        logger.info("DR3查询完成: %d颗参考星", n)
        return ra, dec, mag
    finally:
        client.close()


def get_pixel_scale_from_header(metadata: ImageMetadataPy) -> Optional[float]:
    """
    从FITS头读取像素尺度s0 (arcsec/pixel)。

    优先级:
        1. WCS CD矩阵行列式 → pixel_scale
        2. FOCALLEN + XPIXSZ → s0 = 206.265 * xpixsz_um / focallen_mm

    Args:
        metadata: 图像元数据

    Returns:
        s0 (arcsec/pixel), 无法获取时返回None
    """
    # 1. 尝试WCS pixel_scale
    if metadata.wcs is not None and metadata.wcs.has_wcs:
        ps = metadata.wcs.pixel_scale
        if ps > 0:
            logger.info("从WCS CD矩阵读取像素尺度: s0=%.4f arcsec/pixel", ps)
            return float(ps)

    # 2. 尝试FOCALLEN + XPIXSZ
    obs = metadata.observation
    if obs is not None and obs.focallen is not None and obs.xpixsz is not None:
        if obs.focallen > 0 and obs.xpixsz > 0:
            # s0 = 206.265 * xpixsz(um) / focallen(mm)
            s0 = 206.265 * obs.xpixsz / obs.focallen
            logger.info("从FOCALLEN=%.1fmm + XPIXSZ=%.2fum 计算像素尺度: s0=%.4f arcsec/pixel",
                         obs.focallen, obs.xpixsz, s0)
            return float(s0)

    logger.warning("无法从FITS头读取像素尺度")
    return None


def get_pointing_from_header(metadata: ImageMetadataPy) -> Optional[Tuple[float, float]]:
    """
    从FITS头读取指向(RA,Dec)，用于测试harness查询DR3构建本地索引。

    注意: 此信息仅用于测试harness构建本地索引，不传入匹配算法本身。

    Args:
        metadata: 图像元数据

    Returns:
        (ra, dec) 度, 无法获取时返回None
    """
    if metadata.wcs is not None and metadata.wcs.has_wcs:
        ra0 = metadata.wcs.crval1
        dec0 = metadata.wcs.crval2
        if ra0 != 0.0 or dec0 != 0.0:
            return float(ra0), float(dec0)
    return None
