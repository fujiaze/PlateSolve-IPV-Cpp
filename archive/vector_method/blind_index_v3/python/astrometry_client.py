# -*- coding: utf-8 -*-
"""
Astrometry.net API 客户端
功能: 登录 astrometry.net, 上传 FITS 文件进行 plate solving, 轮询 job 状态, 获取 WCS 结果
用途: DD-SPPS 验证 — 将 astrometry.net 求解结果作为独立参考, 对比 DD-SPPS 求解精度
依赖: requests

API 流程:
    1. POST /api/login (apikey) → session
    2. POST /api/upload (session + file + scale/center 参数) → submission_id
    3. GET /api/submissions/{sub_id} → jobs 列表 (job_id)
    4. GET /api/jobs/{job_id} → status (success/failure/processing)
    5. GET /api/jobs/{job_id}/calibrations/ → WCS (CRVAL, CD, pixscale, orientation)

参数约束 (DD-SPPS 已知 s0/RA/Dec):
    - scale_units='arcsecperpix', scale_type='ul'
    - scale_lower=s0×0.85, scale_upper=s0×1.15  (15% 容差, 适应焦距漂移)
    - center_ra/dec + radius=2.0  (限制搜索区域, 加速)
"""
from __future__ import annotations

import json
import logging
import math
import time
from dataclasses import dataclass, field
from typing import Optional, Tuple

import requests

logger = logging.getLogger("ddspps.astrometry")

_ASTROMETRY_URL = "https://nova.astrometry.net/api"
_ASTROMETRY_UPLOAD_URL = "https://nova.astrometry.net/api/upload"
_ASTROMETRY_LOGIN_URL = "https://nova.astrometry.net/api/login"
_REQUEST_TIMEOUT = 60  # 默认 HTTP 超时 (秒)
_JOB_POLL_INTERVAL = 5.0  # job 状态轮询间隔 (秒)
_JOB_MAX_WAIT = 600  # 单 job 最大等待 (秒)


@dataclass
class AstrometryCalibration:
    """Astrometry.net 求解结果 (校准信息)。"""
    success: bool = False
    job_id: int = -1
    crval1: float = 0.0
    crval2: float = 0.0
    cd11: float = 0.0
    cd12: float = 0.0
    cd21: float = 0.0
    cd22: float = 0.0
    pixscale: float = 0.0  # arcsec/pixel
    orientation: float = 0.0  # 度, 图像 y 轴相对正北方向 (东为正)
    ra_center: float = 0.0  # 图像中心 RA (度)
    dec_center: float = 0.0  # 图像中心 Dec (度)
    width: int = 0  # 像素
    height: int = 0  # 像素
    error: str = ""


class AstrometryClient:
    """Astrometry.net API 客户端。"""

    def __init__(self, api_key: str):
        """
        Args:
            api_key: astrometry.net API key
        """
        self.api_key = api_key
        self.session_id: Optional[str] = None
        self._session = requests.Session()

    def login(self) -> bool:
        """登录获取 session。"""
        try:
            resp = self._session.post(
                _ASTROMETRY_LOGIN_URL,
                data={"request-json": json.dumps({"apikey": self.api_key})},
                timeout=_REQUEST_TIMEOUT,
            )
            resp.raise_for_status()
            data = resp.json()
            if data.get("status") == "success":
                self.session_id = data["session"]
                logger.info("Astrometry.net 登录成功: session=%s", self.session_id[:8])
                return True
            logger.error("Astrometry.net 登录失败: %s", data)
            return False
        except Exception as e:
            logger.error("Astrometry.net 登录异常: %s", e)
            return False

    def upload_fits(
        self,
        fits_path: str,
        s0: float,
        center_ra: Optional[float] = None,
        center_dec: Optional[float] = None,
        radius_deg: float = 2.0,
    ) -> Optional[int]:
        """
        上传 FITS 文件进行 plate solving。

        Args:
            fits_path: FITS 文件路径
            s0: 像素尺度 (arcsec/pixel), 用于 scale 约束
            center_ra, center_dec: 大致指向 (度), 限制搜索区域加速
            radius_deg: 搜索半径 (度)

        Returns:
            submission_id (成功) / None (失败)
        """
        if self.session_id is None:
            logger.error("未登录, 无法上传")
            return None

        # 构建请求参数
        params = {
            "session": self.session_id,
            "allow_commercial_use": "n",
            "publicly_visible": "n",
            "scale_units": "arcsecperpix",
            "scale_type": "ul",
            "scale_lower": s0 * 0.85,
            "scale_upper": s0 * 1.15,
            "downsample_factor": "2",  # 2x 降采样加速 (4096→2048)
        }
        if center_ra is not None and center_dec is not None:
            params["center_ra"] = center_ra
            params["center_dec"] = center_dec
            params["radius"] = radius_deg

        try:
            with open(fits_path, "rb") as f:
                files = {"file": f}
                data = {"request-json": json.dumps(params)}
                resp = self._session.post(
                    _ASTROMETRY_UPLOAD_URL,
                    files=files,
                    data=data,
                    timeout=300,  # 大文件上传 5 分钟超时
                )
            resp.raise_for_status()
            result = resp.json()
            if result.get("status") == "success":
                sub_id = result["subid"]
                logger.info("上传成功: %s, submission_id=%d", fits_path, sub_id)
                return sub_id
            logger.error("上传失败: %s", result)
            return None
        except Exception as e:
            logger.error("上传异常 (%s): %s", fits_path, e)
            return None

    def wait_for_job(
        self, submission_id: int, max_wait: float = _JOB_MAX_WAIT
    ) -> Optional[int]:
        """
        等待 submission 完成, 返回 job_id。

        Args:
            submission_id: submission ID
            max_wait: 最大等待 (秒)

        Returns:
            job_id (成功) / None (失败或超时)
        """
        start = time.time()
        while time.time() - start < max_wait:
            try:
                resp = self._session.get(
                    f"{_ASTROMETRY_URL}/submissions/{submission_id}",
                    timeout=_REQUEST_TIMEOUT,
                )
                resp.raise_for_status()
                data = resp.json()
                jobs = data.get("jobs", [])
                if not jobs:
                    # 还在排队
                    time.sleep(_JOB_POLL_INTERVAL)
                    continue
                # jobs 是 list, 元素可能是 None (processing) 或 int (job_id)
                # 取第一个非 None 的 job
                job = jobs[0]
                if job is None:
                    time.sleep(_JOB_POLL_INTERVAL)
                    continue
                # job 存在, 等待 job 完成
                return self._wait_job_status(int(job), max_wait - (time.time() - start))
            except Exception as e:
                logger.warning("查询 submission %d 异常: %s", submission_id, e)
                time.sleep(_JOB_POLL_INTERVAL)
        logger.error("submission %d 超时", submission_id)
        return None

    def _wait_job_status(
        self, job_id: int, max_wait: float
    ) -> Optional[int]:
        """等待 job 完成, 返回 job_id (成功) / None (失败)。"""
        start = time.time()
        while time.time() - start < max_wait:
            try:
                resp = self._session.get(
                    f"{_ASTROMETRY_URL}/jobs/{job_id}",
                    timeout=_REQUEST_TIMEOUT,
                )
                resp.raise_for_status()
                data = resp.json()
                status = data.get("status", "")
                if status == "success":
                    logger.info("job %d 完成: success", job_id)
                    return job_id
                if status == "failure":
                    logger.warning("job %d 失败: %s", job_id, data)
                    return None
                # processing
                time.sleep(_JOB_POLL_INTERVAL)
            except Exception as e:
                logger.warning("查询 job %d 异常: %s", job_id, e)
                time.sleep(_JOB_POLL_INTERVAL)
        logger.error("job %d 超时", job_id)
        return None

    def get_calibration(self, job_id: int) -> Optional[AstrometryCalibration]:
        """
        获取 job 的 WCS 校准结果。

        方法: 下载 WCS FITS 文件, 直接读取 CD 矩阵 (最准确)
        备用: /api/jobs/{job_id}/calibration/ JSON (只返回 ra/dec/pixscale/orientation/parity,
              不含 CD 矩阵, 从 orientation+parity 重建 CD 会因约定差异引入 90° 偏差)

        WCS 文件 URL: http://nova.astrometry.net/wcs_file/{job_id}
        返回 FITS 文件, header 含 CRVAL1/2, CD1_1/CD1_2/CD2_1/CD2_2, CRPIX1/2
        """
        # 1. 先从 calibration JSON 获取基础信息 (ra/dec/pixscale/orientation)
        try:
            resp = self._session.get(
                f"{_ASTROMETRY_URL}/jobs/{job_id}/calibration/",
                timeout=_REQUEST_TIMEOUT,
            )
            resp.raise_for_status()
            data = resp.json()
            if not isinstance(data, dict) or "ra" not in data:
                logger.warning("job %d 无校准结果: %s", job_id, str(data)[:200])
                return None
            ra = float(data.get("ra", 0.0))
            dec = float(data.get("dec", 0.0))
            pixscale = float(data.get("pixscale", 0.0))
            orientation = float(data.get("orientation", 0.0))
            parity = float(data.get("parity", 1.0))
        except Exception as e:
            logger.error("获取 calibration JSON 异常 (job %d): %s", job_id, e)
            return None

        # 2. 下载 WCS FITS 文件获取准确 CD 矩阵
        cd11 = cd12 = cd21 = cd22 = 0.0
        try:
            wcs_resp = self._session.get(
                f"https://nova.astrometry.net/wcs_file/{job_id}",
                timeout=_REQUEST_TIMEOUT,
            )
            wcs_resp.raise_for_status()
            # 保存到临时文件, 用 ImageReader 解析
            import os
            import tempfile
            from lib.astro_image_io.python.astro_image_io import ImageReader
            tmp_path = os.path.join(tempfile.gettempdir(), f"astrometry_wcs_{job_id}.fits")
            with open(tmp_path, "wb") as f:
                f.write(wcs_resp.content)
            reader = ImageReader()
            img = reader.read_header_only(tmp_path)
            cd11 = float(img.get_keyword_float("CD1_1", 0.0))
            cd12 = float(img.get_keyword_float("CD1_2", 0.0))
            cd21 = float(img.get_keyword_float("CD2_1", 0.0))
            cd22 = float(img.get_keyword_float("CD2_2", 0.0))
            # 也从 WCS 文件读取 CRVAL (更准确)
            ra = float(img.get_keyword_float("CRVAL1", ra))
            dec = float(img.get_keyword_float("CRVAL2", dec))
            img.close()
            os.remove(tmp_path)
            logger.info("job %d WCS 文件: CD=[[%.3e,%.3e],[%.3e,%.3e]]",
                        job_id, cd11, cd12, cd21, cd22)
        except Exception as e:
            logger.warning("下载 WCS 文件失败 (job %d): %s, 用 calibration JSON 重建 CD", job_id, e)
            # 备用: 从 orientation 重建 (可能有 90° 偏差, 仅作 fallback)
            theta_rad = math.radians(orientation)
            s_deg = pixscale / 3600.0
            cd11 = s_deg * math.cos(theta_rad) * parity
            cd12 = -s_deg * math.sin(theta_rad)
            cd21 = s_deg * math.sin(theta_rad) * parity
            cd22 = s_deg * math.cos(theta_rad)

        result = AstrometryCalibration(
            success=True,
            job_id=job_id,
            crval1=ra,
            crval2=dec,
            cd11=cd11,
            cd12=cd12,
            cd21=cd21,
            cd22=cd22,
            pixscale=pixscale,
            orientation=orientation,
            ra_center=ra,
            dec_center=dec,
            width=0,
            height=0,
        )
        logger.info(
            "job %d 校准: CRVAL=(%.6f, %.6f), pixscale=%.4f\"/px, orient=%.3f°, parity=%.1f, θ_cd=%.3f°",
            job_id, result.crval1, result.crval2, result.pixscale, result.orientation, parity,
            math.degrees(math.atan2(cd21, cd22)),
        )
        return result

    def solve(
        self,
        fits_path: str,
        s0: float,
        center_ra: Optional[float] = None,
        center_dec: Optional[float] = None,
        radius_deg: float = 2.0,
        max_wait: float = _JOB_MAX_WAIT,
    ) -> Optional[AstrometryCalibration]:
        """
        完整流程: 上传 → 等待 → 获取 WCS。

        Args:
            fits_path: FITS 文件路径
            s0: 像素尺度 (arcsec/pixel)
            center_ra, center_dec: 大致指向 (度)
            radius_deg: 搜索半径 (度)
            max_wait: 最大等待 (秒)

        Returns:
            AstrometryCalibration (成功) / None (失败)
        """
        sub_id = self.upload_fits(
            fits_path, s0, center_ra, center_dec, radius_deg
        )
        if sub_id is None:
            return None
        job_id = self.wait_for_job(sub_id, max_wait)
        if job_id is None:
            return None
        return self.get_calibration(job_id)
