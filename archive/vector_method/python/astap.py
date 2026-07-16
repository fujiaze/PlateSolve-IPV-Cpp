from __future__ import annotations

import os
import subprocess
import tempfile
from dataclasses import dataclass, field
from typing import Optional, Tuple


@dataclass
class AstapResult:
    success: bool = False
    ra_deg: float = 0.0
    dec_deg: float = 0.0
    scale_arcsec_px: float = 0.0
    rotation_deg: float = 0.0
    matched_stars: int = 0
    rms_arcsec: float = 0.0
    error_message: str = ""


class AstapSolver:
    def __init__(self, astap_path: str = "C:\\Program Files\\astap\\astap.exe"):
        self.astap_path = astap_path
        self._validate_path()

    def _validate_path(self):
        if not os.path.exists(self.astap_path):
            raise FileNotFoundError(f"ASTAP not found at: {self.astap_path}")

    def solve(
        self,
        image_path: str,
        ra_hint_deg: Optional[float] = None,
        dec_hint_deg: Optional[float] = None,
        scale_hint_arcsec_px: Optional[float] = None,
        timeout_seconds: int = 60,
        max_stars: int = 500
    ) -> AstapResult:
        if not os.path.exists(image_path):
            return AstapResult(success=False, error_message=f"图像文件不存在: {image_path}")

        with tempfile.NamedTemporaryFile(mode="w", suffix=".ini", delete=False, encoding="utf-8") as ini_file:
            ini_path = ini_file.name
            ini_content = self._generate_ini_content(image_path, ra_hint_deg, dec_hint_deg, scale_hint_arcsec_px, max_stars)
            ini_file.write(ini_content)

        with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False, encoding="utf-8") as result_file:
            result_path = result_file.name

        try:
            cmd = [
                self.astap_path,
                f"-f{image_path}",
                f"-o{result_path}",
                f"-t{timeout_seconds}"
            ]

            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=timeout_seconds + 10,
                encoding="utf-8",
                errors="replace"
            )

            if result.returncode != 0:
                return AstapResult(success=False, error_message=f"ASTAP执行失败: {result.stderr}")

            return self._parse_result(result_path)

        except subprocess.TimeoutExpired:
            return AstapResult(success=False, error_message="ASTAP超时")
        except Exception as e:
            return AstapResult(success=False, error_message=f"ASTAP调用异常: {str(e)}")
        finally:
            if os.path.exists(ini_path):
                os.unlink(ini_path)
            if os.path.exists(result_path):
                os.unlink(result_path)

    def _generate_ini_content(
        self,
        image_path: str,
        ra_hint_deg: Optional[float],
        dec_hint_deg: Optional[float],
        scale_hint_arcsec_px: Optional[float],
        max_stars: int
    ) -> str:
        lines = []
        lines.append(f"File={image_path}")
        lines.append(f"Maxstars={max_stars}")
        
        if ra_hint_deg is not None and dec_hint_deg is not None:
            lines.append(f"RA={ra_hint_deg}")
            lines.append(f"DEC={dec_hint_deg}")
            if scale_hint_arcsec_px is not None:
                lines.append(f"Scale={scale_hint_arcsec_px}")
        
        return "\n".join(lines) + "\n"

    def _parse_result(self, result_path: str) -> AstapResult:
        result = AstapResult()

        if not os.path.exists(result_path):
            result.error_message = "结果文件未生成"
            return result

        try:
            with open(result_path, "r", encoding="utf-8") as f:
                content = f.read()

            lines = content.split("\n")
            for line in lines:
                line = line.strip()
                if line.startswith("RA"):
                    parts = line.split("=")
                    if len(parts) >= 2:
                        result.ra_deg = float(parts[1].strip())
                elif line.startswith("DEC"):
                    parts = line.split("=")
                    if len(parts) >= 2:
                        result.dec_deg = float(parts[1].strip())
                elif line.startswith("Scale"):
                    parts = line.split("=")
                    if len(parts) >= 2:
                        result.scale_arcsec_px = float(parts[1].strip())
                elif line.startswith("Rotation"):
                    parts = line.split("=")
                    if len(parts) >= 2:
                        result.rotation_deg = float(parts[1].strip())
                elif line.startswith("Stars"):
                    parts = line.split("=")
                    if len(parts) >= 2:
                        result.matched_stars = int(parts[1].strip())
                elif line.startswith("RMS"):
                    parts = line.split("=")
                    if len(parts) >= 2:
                        result.rms_arcsec = float(parts[1].strip())

            result.success = result.matched_stars > 0 and result.ra_deg != 0.0
            if not result.success:
                result.error_message = "解析未成功匹配足够的星点"

        except Exception as e:
            result.error_message = f"结果解析失败: {str(e)}"

        return result

    def blind_solve(self, image_path: str, timeout_seconds: int = 60) -> AstapResult:
        return self.solve(image_path, timeout_seconds=timeout_seconds)

    def solve_with_hint(
        self,
        image_path: str,
        ra_deg: float,
        dec_deg: float,
        timeout_seconds: int = 60
    ) -> AstapResult:
        return self.solve(image_path, ra_hint_deg=ra_deg, dec_hint_deg=dec_deg, timeout_seconds=timeout_seconds)