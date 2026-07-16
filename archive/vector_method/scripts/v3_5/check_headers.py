"""检查所有测试帧的FITS头信息，筛选有标准RA/DEC/FOCALLEN/XPIXSZ的帧"""
import sys, os, glob
if sys.platform == 'win32':
    import locale; locale.setlocale(locale.LC_ALL, 'en_US.UTF-8')

PROJECT_ROOT = r'F:\Astro dev\Astro CS Normalization Database'
testdata = os.path.join(PROJECT_ROOT, 'testdata')

from astropy.io import fits as afits

# 收集所有fts文件
all_files = glob.glob(os.path.join(testdata, '**', '*.fts'), recursive=True)
print(f'共找到 {len(all_files)} 个FITS文件\n')

valid = []
invalid = []

for f in sorted(all_files):
    try:
        hdul = afits.open(f)
        hdr = hdul[0].header
        hdul.close()
    except Exception as e:
        invalid.append((f, f'读取失败: {e}'))
        continue

    ra = hdr.get('RA', hdr.get('OBJCTRA', None))
    dec = hdr.get('DEC', hdr.get('OBJCTDEC', None))
    fl = hdr.get('FOCALLEN', hdr.get('FOCLENGTH', hdr.get('TELESCOP', None)))
    ps = hdr.get('XPIXSZ', hdr.get('PIXSIZE', hdr.get('CDELT1', None)))

    # 检查关键字段
    missing = []
    if ra is None: missing.append('RA')
    if dec is None: missing.append('DEC')
    if fl is None: missing.append('FOCALLEN')
    if ps is None: missing.append('XPIXSZ')

    basename = os.path.basename(f)
    # 提取目标名和滤镜
    parts = basename.split('_')
    target = parts[0] if len(parts) > 0 else '?'

    if missing:
        invalid.append((basename, f'缺少: {", ".join(missing)}'))
    else:
        valid.append((basename, f'RA={ra} DEC={dec} FL={fl} PS={ps}'))

print(f'=== 有效帧 ({len(valid)}) ===')
for name, info in valid:
    print(f'  {name}: {info}')

print(f'\n=== 无效帧 ({len(invalid)}) ===')
for name, info in invalid:
    print(f'  {name}: {info}')
