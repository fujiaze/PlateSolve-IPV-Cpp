from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class AstapConfig:
    enabled: bool = True
    path: str = "C:\\Program Files\\astap\\astap.exe"
    max_stars: int = 500
    timeout_seconds: int = 60


@dataclass
class GaiaConfig:
    data_dir: str = ""
    max_stars: int = 2000
    mag_limit: float = 14.0


@dataclass
class SolverConfig:
    use_header_wcs: bool = True
    blind_solve_fallback: bool = True
    min_matches: int = 6
    max_iterations: int = 5
    rms_threshold_pixels: float = 0.5


@dataclass
class PlateSolveConfig:
    astap: AstapConfig = field(default_factory=AstapConfig)
    gaia: GaiaConfig = field(default_factory=GaiaConfig)
    solver: SolverConfig = field(default_factory=SolverConfig)

    @classmethod
    def from_file(cls, config_path: Optional[str] = None) -> "PlateSolveConfig":
        if config_path is None:
            base = os.path.dirname(os.path.abspath(__file__))
            config_path = os.path.normpath(os.path.join(base, "..", "config", "plate_solve_config.json"))
        
        if os.path.exists(config_path):
            try:
                with open(config_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                
                return cls(
                    astap=AstapConfig(**data.get("astap", {})),
                    gaia=GaiaConfig(**data.get("gaia", {})),
                    solver=SolverConfig(**data.get("solver", {}))
                )
            except Exception as e:
                print(f"配置文件读取失败: {e}")
        
        return cls()

    def save(self, config_path: Optional[str] = None):
        if config_path is None:
            base = os.path.dirname(os.path.abspath(__file__))
            config_path = os.path.normpath(os.path.join(base, "..", "config", "plate_solve_config.json"))
        
        config_dir = os.path.dirname(config_path)
        if not os.path.exists(config_dir):
            os.makedirs(config_dir, exist_ok=True)
        
        data = {
            "astap": {
                "enabled": self.astap.enabled,
                "path": self.astap.path,
                "max_stars": self.astap.max_stars,
                "timeout_seconds": self.astap.timeout_seconds
            },
            "gaia": {
                "data_dir": self.gaia.data_dir,
                "max_stars": self.gaia.max_stars,
                "mag_limit": self.gaia.mag_limit
            },
            "solver": {
                "use_header_wcs": self.solver.use_header_wcs,
                "blind_solve_fallback": self.blind_solve_fallback,
                "min_matches": self.solver.min_matches,
                "max_iterations": self.solver.max_iterations,
                "rms_threshold_pixels": self.solver.rms_threshold_pixels
            }
        }
        
        with open(config_path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)