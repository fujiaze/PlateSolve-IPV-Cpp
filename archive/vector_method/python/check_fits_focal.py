#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""检查FITS头中的焦距信息"""
import os
import sys

project_root = r"F:\Astro dev\Astro CS Normalization Database"
sys.path.insert(0, project_root)

from lib.astro_image_io.python.astro_image_io import ImageReader

test_dir = os.path.join(project_root, 'testdata', 'lights', 'panel1')
files = [f for f in os.listdir(test_dir) if f.endswith('.fts')][:5]

print('检查FITS头中的焦距信息:')
print('=' * 70)

reader = ImageReader()
for f in files:
    path = os.path.join(test_dir, f)
    with reader.read(path) as img:
        print(f'\n文件: {f}')
        print(f'  FOCALLEN: {img.get_keyword("FOCALLEN")}')
        print(f'  FOCAL: {img.get_keyword("FOCAL")}')
        print(f'  SCALE: {img.get_keyword("SCALE")}')
        print(f'  CDELT1: {img.get_keyword("CDELT1")}')
        print(f'  CDELT2: {img.get_keyword("CDELT2")}')
        print(f'  CD1_1: {img.get_keyword("CD1_1")}')
        print(f'  pixel_scale: {img.wcs.pixel_scale:.4f} "/px')
