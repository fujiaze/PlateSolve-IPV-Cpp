"""
ASTAP Plate Solver - ASTAP解析器封装
功能: 调用ASTAP进行天文图像Plate Solve解析
用途: 提供初始位置提示或盲解析，返回WCS解
参考: https://www.hnsky.org/astap.htm

命令行参数说明:
-f filename        输入文件路径
-ra hours          中心RA（小时格式，如18.187）
-spd degrees       中心SPD（南极度数 = DEC + 90）
-r degrees         搜索半径（度）
-fov degrees       FOV高度（度）
-s number          最大星数（典型值500）
-t tolerance       容差
-m pixels          最小星尺寸
-d path            星表数据库路径
-speed auto/slow   速度模式
-update            更新FITS文件头
-log               写入日志文件
-wcs               写入WCS文件
"""

from __future__ import annotations

import os
import subprocess
import time
from dataclasses import dataclass
from typing import Optional, Tuple, List
import math


@dataclass
class AstapResult:
    success: bool = False
    ra_deg: float = 0.0
    dec_deg: float = 0.0
    ra_hms: str = ""
    dec_dms: str = ""
    scale_arcsec_px: float = 0.0
    rotation_deg: float = 0.0
    matched_stars: int = 0
    rms_arcsec: float = 0.0
    solve_time_sec: float = 0.0
    error_message: str = ""
    database_used: str = ""


@dataclass
class AstapConfig:
    astap_path: str = r"C:\Program Files\astap\astap.exe"
    database_dir: str = r"C:\Program Files\astap"
    max_stars: int = 5000
    search_radius_deg: float = 10.0
    tolerance: float = 0.01
    min_star_size: float = 2.0
    timeout_seconds: int = 120
    speed_mode: str = "slow"
    fov_multiplier: float = 3.0
    retry_enabled: bool = True
    retry_max_attempts: int = 5
    retry_star_size_step: float = 0.3
    retry_star_size_max: float = 5.0


class AstapSolver:
    def __init__(self, config: Optional[AstapConfig] = None):
        self.config = config if config else AstapConfig()
        self._validate_paths()
    
    def _validate_paths(self):
        if not os.path.exists(self.config.astap_path):
            raise FileNotFoundError(f"ASTAP not found: {self.config.astap_path}")
        if not os.path.isdir(self.config.database_dir):
            raise FileNotFoundError(f"Database directory not found: {self.config.database_dir}")
    
    def _calculate_fov(self, focal_length_mm: float, pixel_size_um: float, width: int, height: int) -> Tuple[float, float, float]:
        """
        计算FOV
        返回: (fov_width_deg, fov_height_deg, fov_diagonal_deg)
        """
        if focal_length_mm <= 0 or pixel_size_um <= 0:
            return (0.0, 0.0, 0.0)
        
        scale_arcsec_px = 206.265 * pixel_size_um / focal_length_mm
        scale_deg_px = scale_arcsec_px / 3600.0
        
        fov_w = width * scale_deg_px
        fov_h = height * scale_deg_px
        fov_diag = math.sqrt(fov_w**2 + fov_h**2)
        
        return (fov_w, fov_h, fov_diag)
    
    def _select_database(self, fov_diagonal_deg: float) -> str:
        """
        根据FOV选择合适的星表数据库
        G05: 用于大视场 (>5°)
        D05/D20/D50/D80: 用于小视场
        """
        db_dir = self.config.database_dir
        
        if fov_diagonal_deg >= 5.0:
            db_name = "g05"
        elif fov_diagonal_deg >= 2.0:
            db_name = "d05"
        elif fov_diagonal_deg >= 1.0:
            db_name = "d20"
        else:
            db_name = "d50"
        
        for f in os.listdir(db_dir):
            if f.lower().startswith(db_name) and f.endswith('.1476'):
                return os.path.join(db_dir, f)
        
        return db_dir
    
    def _ra_deg_to_hours(self, ra_deg: float) -> float:
        return ra_deg / 15.0
    
    def _dec_to_spd(self, dec_deg: float) -> float:
        return dec_deg + 90.0
    
    def _deg_to_hms(self, deg: float) -> str:
        hours = deg / 15.0
        h = int(hours)
        m = int((hours - h) * 60)
        s = (hours - h - m/60) * 3600
        return f"{h:02d}:{m:02d}:{s:05.2f}"
    
    def _deg_to_dms(self, deg: float) -> str:
        sign = '+' if deg >= 0 else '-'
        deg = abs(deg)
        d = int(deg)
        m = int((deg - d) * 60)
        s = (deg - d - m/60) * 3600
        return f"{sign}{d:02d}:{m:02d}:{s:04.1f}"
    
    def solve(
        self,
        image_path: str,
        ra_hint_deg: Optional[float] = None,
        dec_hint_deg: Optional[float] = None,
        focal_length_mm: Optional[float] = None,
        pixel_size_um: Optional[float] = None,
        width: Optional[int] = None,
        height: Optional[int] = None,
        search_radius_deg: Optional[float] = None,
        max_stars: Optional[int] = None,
        min_star_size: Optional[float] = None
    ) -> AstapResult:
        """
        执行ASTAP解析，支持重试机制
        
        参数:
            image_path: 图像文件路径
            ra_hint_deg: RA提示（度），可选
            dec_hint_deg: DEC提示（度），可选
            focal_length_mm: 焦距（mm）
            pixel_size_um: 像元尺寸（um）
            width: 图像宽度
            height: 图像高度
            search_radius_deg: 搜索半径（度）
            max_stars: 最大星数
            min_star_size: 最小星尺寸（像素）
        """
        if not os.path.exists(image_path):
            return AstapResult(success=False, error_message=f"图像文件不存在: {image_path}")
        
        current_min_star_size = min_star_size if min_star_size else self.config.min_star_size
        attempt = 0
        last_result = None
        
        while True:
            attempt += 1
            print(f"  尝试 {attempt}: min_star_size={current_min_star_size:.1f}")
            
            result = self._solve_single(
                image_path=image_path,
                ra_hint_deg=ra_hint_deg,
                dec_hint_deg=dec_hint_deg,
                focal_length_mm=focal_length_mm,
                pixel_size_um=pixel_size_um,
                width=width,
                height=height,
                search_radius_deg=search_radius_deg,
                max_stars=max_stars,
                min_star_size=current_min_star_size
            )
            
            last_result = result
            
            if result.success:
                if attempt > 1:
                    print(f"  重试成功！共尝试 {attempt} 次")
                return result
            
            if not self.config.retry_enabled:
                return result
            
            if attempt >= self.config.retry_max_attempts:
                print(f"  达到最大重试次数 {self.config.retry_max_attempts}，放弃")
                return result
            
            next_min_star_size = current_min_star_size + self.config.retry_star_size_step
            if next_min_star_size > self.config.retry_star_size_max:
                print(f"  min_star_size 达到上限 {self.config.retry_star_size_max}，放弃")
                return result
            
            current_min_star_size = next_min_star_size
        
        return last_result
    
    def _solve_single(
        self,
        image_path: str,
        ra_hint_deg: Optional[float] = None,
        dec_hint_deg: Optional[float] = None,
        focal_length_mm: Optional[float] = None,
        pixel_size_um: Optional[float] = None,
        width: Optional[int] = None,
        height: Optional[int] = None,
        search_radius_deg: Optional[float] = None,
        max_stars: Optional[int] = None,
        min_star_size: Optional[float] = None
    ) -> AstapResult:
        """单次ASTAP解析调用"""
        cmd = [self.config.astap_path, "-f", image_path]
        
        fov_diag = 0.0
        if focal_length_mm and pixel_size_um and width and height:
            fov_w, fov_h, fov_diag = self._calculate_fov(focal_length_mm, pixel_size_um, width, height)
            print(f"  FOV: {fov_w:.2f}° x {fov_h:.2f}° (对角线: {fov_diag:.2f}°)")
            
            fov_height_deg = fov_h
            cmd.extend(["-fov", f"{fov_height_deg:.4f}"])
        
        db_path = self._select_database(fov_diag)
        cmd.extend(["-d", db_path])
        
        if ra_hint_deg is not None and dec_hint_deg is not None:
            ra_hours = self._ra_deg_to_hours(ra_hint_deg)
            spd = self._dec_to_spd(dec_hint_deg)
            cmd.extend(["-ra", f"{ra_hours:.8f}"])
            cmd.extend(["-spd", f"{spd:.6f}"])
            print(f"  位置提示: RA={ra_hint_deg:.4f}° ({ra_hours:.6f}h), DEC={dec_hint_deg:.4f}° (SPD={spd:.4f}°)")
        
        radius = search_radius_deg if search_radius_deg else self.config.search_radius_deg
        if fov_diag > 0:
            radius = max(radius, fov_diag * self.config.fov_multiplier)
        cmd.extend(["-r", f"{radius:.2f}"])
        
        stars = max_stars if max_stars else self.config.max_stars
        cmd.extend(["-s", str(stars)])
        
        current_min_star_size = min_star_size if min_star_size else self.config.min_star_size
        if current_min_star_size > 0:
            cmd.extend(["-m", f"{current_min_star_size:.1f}"])
        
        cmd.extend(["-t", str(self.config.tolerance)])
        cmd.extend(["-speed", self.config.speed_mode])
        cmd.extend(["-update", "-log"])
        
        print(f"  命令: {' '.join(cmd)}")
        
        start_time = time.time()
        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=self.config.timeout_seconds + 10,
                encoding="utf-8",
                errors="replace"
            )
            solve_time = time.time() - start_time
            
            return self._parse_result(image_path, solve_time)
            
        except subprocess.TimeoutExpired:
            return AstapResult(success=False, error_message="ASTAP超时")
        except Exception as e:
            return AstapResult(success=False, error_message=f"ASTAP调用异常: {str(e)}")
    
    def _parse_result(self, image_path: str, solve_time: float) -> AstapResult:
        ini_path = image_path.rsplit('.', 1)[0] + '.ini'
        log_path = image_path.rsplit('.', 1)[0] + '.log'
        
        astap_result = AstapResult()
        astap_result.solve_time_sec = solve_time
        
        if not os.path.exists(ini_path):
            astap_result.error_message = "INI结果文件未生成"
            return astap_result
        
        try:
            with open(ini_path, 'r', encoding='utf-8') as f:
                ini_content = f.read()
            
            lines = ini_content.split('\n')
            for line in lines:
                line = line.strip()
                if line.startswith('PLTSOLVD='):
                    val = line.split('=')[1].strip()
                    astap_result.success = (val == 'T')
                elif line.startswith('CRVAL1'):
                    val = line.split('=')[1].strip().split()[0]
                    astap_result.ra_deg = float(val)
                    astap_result.ra_hms = self._deg_to_hms(astap_result.ra_deg)
                elif line.startswith('CRVAL2'):
                    val = line.split('=')[1].strip().split()[0]
                    astap_result.dec_deg = float(val)
                    astap_result.dec_dms = self._deg_to_dms(astap_result.dec_deg)
                elif line.startswith('CDELT1'):
                    val = line.split('=')[1].strip().split()[0]
                    scale_deg = abs(float(val))
                    astap_result.scale_arcsec_px = scale_deg * 3600.0
                elif line.startswith('CROTA1'):
                    val = line.split('=')[1].strip().split()[0]
                    astap_result.rotation_deg = float(val)
            
            if os.path.exists(log_path):
                with open(log_path, 'r', encoding='utf-8') as f:
                    log_content = f.read()
                
                for line in log_content.split('\n'):
                    if 'Using star database' in line:
                        parts = line.split('Using star database')
                        if len(parts) > 1:
                            astap_result.database_used = parts[1].strip()
                    if 'Solution found' in line:
                        if 'Used stars down to magnitude' in line:
                            parts = line.split('magnitude:')
                            if len(parts) > 1:
                                try:
                                    astap_result.matched_stars = int(float(parts[1].strip().split()[0]))
                                except:
                                    pass
            
            if not astap_result.success:
                if os.path.exists(log_path):
                    with open(log_path, 'r', encoding='utf-8') as f:
                        log_lines = f.readlines()
                    for line in reversed(log_lines):
                        if 'No solution found' in line:
                            astap_result.error_message = "ASTAP未找到解"
                            break
            
        except Exception as e:
            astap_result.error_message = f"结果解析失败: {str(e)}"
        
        return astap_result
    
    def blind_solve(self, image_path: str, **kwargs) -> AstapResult:
        return self.solve(image_path, **kwargs)
    
    def solve_with_hint(
        self,
        image_path: str,
        ra_deg: float,
        dec_deg: float,
        **kwargs
    ) -> AstapResult:
        return self.solve(image_path, ra_hint_deg=ra_deg, dec_hint_deg=dec_deg, **kwargs)


def test_astap():
    print("=== ASTAP 解析测试 ===\n")
    
    config = AstapConfig(
        max_stars=2000,
        search_radius_deg=10.0,
        speed_mode="slow",
        min_star_size=1.5,
        fov_multiplier=3.0
    )
    
    solver = AstapSolver(config)
    
    test_files = [
        {
            "path": r"F:\Astro dev\Astro CS Normalization Database\testdata\lights\panel1\Galaxy_Center_mosaic1_T4_flying_dutchman-20250702@061703-180S-Red.fts",
            "ra_hint": None,
            "dec_hint": None,
            "focal_length": 200.0,
            "pixel_size": 6.0,
            "width": 4500,
            "height": 3600,
            "desc": "盲解析"
        },
        {
            "path": r"F:\Astro dev\Astro CS Normalization Database\testdata\lights\panel1\Galaxy_Center_mosaic1_T4_flying_dutchman-20250702@061703-180S-Red.fts",
            "ra_hint": 272.79,
            "dec_hint": -13.18,
            "focal_length": 200.0,
            "pixel_size": 6.0,
            "width": 4500,
            "height": 3600,
            "desc": "有位置提示"
        },
        {
            "path": r"F:\Astro dev\Astro CS Normalization Database\testdata\lights\panel3\Galaxy_Center_mosaic3_T4_flying_dutchman-20250718@003945-180S-Red.fts",
            "ra_hint": None,
            "dec_hint": None,
            "focal_length": 200.0,
            "pixel_size": 6.0,
            "width": 4500,
            "height": 3600,
            "desc": "盲解析"
        },
        {
            "path": r"F:\Astro dev\Astro CS Normalization Database\testdata\lights\panel3\Galaxy_Center_mosaic3_T4_flying_dutchman-20250718@003945-180S-Red.fts",
            "ra_hint": 272.90,
            "dec_hint": -23.30,
            "focal_length": 200.0,
            "pixel_size": 6.0,
            "width": 4500,
            "height": 3600,
            "desc": "有位置提示"
        }
    ]
    
    for i, test in enumerate(test_files):
        print(f"\n--- 测试 {i+1}: {test.get('desc', '')} ---")
        print(f"    文件: {os.path.basename(test['path'])}")
        
        result = solver.solve(
            image_path=test["path"],
            ra_hint_deg=test.get("ra_hint"),
            dec_hint_deg=test.get("dec_hint"),
            focal_length_mm=test.get("focal_length"),
            pixel_size_um=test.get("pixel_size"),
            width=test.get("width"),
            height=test.get("height")
        )
        
        print(f"\n  结果:")
        print(f"    成功: {result.success}")
        if result.success:
            print(f"    RA: {result.ra_deg:.6f}° ({result.ra_hms})")
            print(f"    DEC: {result.dec_deg:.6f}° ({result.dec_dms})")
            print(f"    Scale: {result.scale_arcsec_px:.4f}\"/px")
            print(f"    Rotation: {result.rotation_deg:.2f}°")
            print(f"    数据库: {result.database_used}")
            print(f"    解析时间: {result.solve_time_sec:.2f}s")
        else:
            print(f"    错误: {result.error_message}")


if __name__ == "__main__":
    test_astap()
