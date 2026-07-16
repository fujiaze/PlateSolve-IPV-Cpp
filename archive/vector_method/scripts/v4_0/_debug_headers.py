"""检查FITS头中的RA/DEC相关关键字"""
import os, sys
PROJECT_ROOT = r"F:\Astro dev\Astro CS Normalization Database"
sys.path.insert(0, os.path.join(PROJECT_ROOT, "lib", "astro_image_io", "python"))
from astro_image_io import ImageReader

# 检查多个文件
files = [
    ("M20_T2", os.path.join(PROJECT_ROOT, "testdata", "lights", "M20_T2_flying_dutchman-20250701@073331-300S-Red.fts")),
    ("NGC4945", os.path.join(PROJECT_ROOT, "testdata", "lights", "T3", "lum", "NGC4945_FD_T3_flying_dutchman-20250205@074722-600S-Lum.fts")),
]

reader = ImageReader()
for name, path in files:
    if not os.path.exists(path):
        print(f"{name}: 文件不存在 {path}")
        continue
    print(f"\n=== {name} ===")
    img = reader.read(path)
    # 打印所有FITS关键字中RA/DEC相关的
    kws = img.keywords
    ra_related = [k for k in kws if any(x in k.name.upper() for x in ['RA', 'DEC', 'OBJCT', 'CRVAL', 'CENTER', 'TEL'])]
    for k in ra_related:
        print(f"  {k.name} = {k.value}  / {k.comment}")
    print(f"\n  focallen={img.metadata.observation.focallen}")
    print(f"  xpixsz={img.metadata.observation.xpixsz}")
    print(f"  has_wcs={img.has_wcs}")
    if img.has_wcs:
        print(f"  crval1={img.metadata.wcs.crval1}")
        print(f"  crval2={img.metadata.wcs.crval2}")
