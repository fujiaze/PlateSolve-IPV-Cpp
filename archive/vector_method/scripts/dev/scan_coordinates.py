"""
扫描testdata所有FITS文件，检查文件头是否有有效坐标(RA/Dec)
输出: 有坐标的文件列表、无坐标的文件列表
"""
import os, sys
import numpy as np

PROJECT_ROOT = r"F:\Astro dev\Astro CS Normalization Database"
sys.path.insert(0, os.path.join(PROJECT_ROOT, "lib", "astro_image_io", "python"))

from astro_image_io import ImageReader


def extract_center_from_keywords(keywords):
    """从FITS关键词提取中心坐标"""
    ra, dec = 0.0, 0.0
    for kw in keywords:
        name = kw.name.upper()
        if name in ("OBJCTRA", "RA"):
            val = kw.value
            if isinstance(val, str):
                parts = val.replace("h", " ").replace("m", " ").replace("s", " ").split()
                if len(parts) >= 3:
                    try:
                        ra = (float(parts[0]) + float(parts[1]) / 60 + float(parts[2]) / 3600) * 15
                    except ValueError:
                        pass
            elif isinstance(val, (int, float)):
                ra = float(val)
        elif name in ("OBJCTDEC", "DEC"):
            val = kw.value
            if isinstance(val, str):
                parts = val.replace("d", " ").replace("'", " ").replace('"', " ").split()
                if len(parts) >= 3:
                    try:
                        sign = -1 if parts[0].startswith("-") else 1
                        dec = sign * (abs(float(parts[0])) + float(parts[1]) / 60 + float(parts[2]) / 3600)
                    except ValueError:
                        pass
            elif isinstance(val, (int, float)):
                dec = float(val)
    return ra, dec


def scan_frames(root_dir):
    """递归扫描所有 .fit/.fits/.fts 文件"""
    frames = []
    for dirpath, dirnames, filenames in os.walk(root_dir):
        for f in filenames:
            if f.lower().endswith(('.fit', '.fits', '.fts')):
                frames.append(os.path.join(dirpath, f))
    return sorted(frames)


def main():
    testdata_dir = os.path.join(PROJECT_ROOT, "testdata")
    frames = scan_frames(testdata_dir)
    print(f"扫描到 {len(frames)} 帧")

    reader = ImageReader()
    valid = []
    invalid = []

    for i, path in enumerate(frames):
        fname = os.path.basename(path)
        try:
            img = reader.read(path)
            ra, dec = 0.0, 0.0

            # 优先从WCS获取
            if img.metadata.wcs and img.metadata.wcs.has_wcs:
                ra = img.metadata.wcs.crval1
                dec = img.metadata.wcs.crval2

            # 从关键词获取
            if ra == 0.0 and dec == 0.0:
                ra, dec = extract_center_from_keywords(img.keywords)

            has_coord = (ra != 0.0 or dec != 0.0)

            if has_coord:
                valid.append(path)
            else:
                invalid.append((path, "RA=0 Dec=0"))

        except Exception as e:
            invalid.append((path, f"读取失败: {str(e)[:50]}"))

        if (i + 1) % 50 == 0:
            print(f"  进度: {i+1}/{len(frames)} (有效={len(valid)} 无效={len(invalid)})")

    print(f"\n结果:")
    print(f"  有效(有坐标): {len(valid)} 帧")
    print(f"  无效(无坐标): {len(invalid)} 帧")

    if invalid:
        print(f"\n无效文件列表:")
        for path, reason in invalid:
            print(f"  {os.path.basename(path)}: {reason}")

    # 保存有效文件列表
    valid_list_path = os.path.join(PROJECT_ROOT, "testdata_valid_frames.txt")
    with open(valid_list_path, 'w', encoding='utf-8') as f:
        for p in valid:
            f.write(p + '\n')
    print(f"\n有效文件列表已保存: {valid_list_path}")


if __name__ == '__main__':
    main()
