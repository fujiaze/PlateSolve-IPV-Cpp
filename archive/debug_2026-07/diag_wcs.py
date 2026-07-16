import sys, numpy as np
from astropy.io import fits
from astropy.coordinates import angular_separation
import astropy.wcs as pywcs

fits_path = 'testdata/results/Galaxy_Center_T4/panel1/Red/Galaxy_Center_mosaic1_T4_flying_dutchman-20250702@061703-180S-Red/01_calibrated.fits'

with fits.open(fits_path) as hdul:
    hdr = hdul[0].header
    img_w = hdr['NAXIS1']
    img_h = hdr['NAXIS2']
    print(f'Image: {img_w}x{img_h}')
    print(f'CRVAL: ({hdr["CRVAL1"]}, {hdr["CRVAL2"]})')
    print(f'CRPIX: ({hdr["CRPIX1"]}, {hdr["CRPIX2"]})')
    cd11 = hdr['CD1_1']
    cd12 = hdr['CD1_2']
    cd21 = hdr['CD2_1']
    cd22 = hdr['CD2_2']
    print(f'CD: [[{cd11}, {cd12}], [{cd21}, {cd22}]]')

    w = pywcs.WCS(hdr)

    # 四角
    corners_px = np.array([[0,0], [img_w-1,0], [0,img_h-1], [img_w-1,img_h-1]], dtype=float)
    corners_world = w.pixel_to_world(corners_px[:,0], corners_px[:,1])

    # 中心
    center_world = w.pixel_to_world(np.array([img_w/2.0]), np.array([img_h/2.0]))
    cra = center_world[0].ra.deg
    cdec = center_world[0].dec.deg
    print(f'Center sky: RA={cra:.6f}, DEC={cdec:.6f}')

    for i in range(4):
        sep = angular_separation(
            cra * np.pi / 180, cdec * np.pi / 180,
            corners_world[i].ra.deg * np.pi / 180,
            corners_world[i].dec.deg * np.pi / 180)
        print(f'  Corner ({corners_px[i][0]:.0f},{corners_px[i][1]:.0f}): '
              f'RA={corners_world[i].ra.deg:.6f} DEC={corners_world[i].dec.deg:.6f} '
              f'sep={sep*180/np.pi:.4f} deg')

    cd = np.array([[cd11, cd12], [cd21, cd22]])
    ps = np.sqrt(abs(np.linalg.det(cd)))
    print(f'Pixel scale: {ps:.6e} deg/px = {ps*3600:.2f} arcsec/px')
    print(f'FOV: {img_w*ps:.2f} x {img_h*ps:.2f} deg')
    print(f'Diagonal: {np.sqrt(img_w**2+img_h**2)*ps:.2f} deg')

    # Gaia 星投影分布
    sys.path.insert(0, 'lib/photometric_calib/spectrum_integrator/python')
    from gaia_spectrum_client import GaiaSpectrumClient
    client = GaiaSpectrumClient(data_dir='GaiaDR3SP')

    max_sep = max(angular_separation(
        cra * np.pi / 180, cdec * np.pi / 180,
        corners_world[i].ra.deg * np.pi / 180,
        corners_world[i].dec.deg * np.pi / 180) for i in range(4))
    cone_radius = max_sep * 180 / np.pi * 1.05
    print(f'Cone radius: {cone_radius:.4f} deg')

    gaia_stars = client.cone_search_with_spectrum(cra, cdec, cone_radius, 8.0, 12.0)
    print(f'Gaia stars: {len(gaia_stars)}')

    # 投影到像素
    ra_arr = np.array([s.ra for s in gaia_stars])
    dec_arr = np.array([s.dec for s in gaia_stars])
    px_arr, py_arr = w.world_to_pixel_values(ra_arr, dec_arr)

    in_img = (px_arr >= -50) & (px_arr < img_w + 50) & (py_arr >= -50) & (py_arr < img_h + 50)
    n_in = int(np.sum(in_img))
    print(f'In image: {n_in}')

    px = px_arr[in_img]
    py = py_arr[in_img]

    # 5x5 grid
    gx = np.clip((px / img_w * 5).astype(int), 0, 4)
    gy = np.clip((py / img_h * 5).astype(int), 0, 4)
    grid = np.zeros((5, 5), dtype=int)
    for i in range(n_in):
        grid[gy[i], gx[i]] += 1
    print('5x5 grid (Gaia stars projected):')
    for row in grid:
        print(f'  {list(row)}')

    # 距中心
    dx = (px - img_w / 2) / (img_w / 2)
    dy = (py - img_h / 2) / (img_h / 2)
    dist = np.sqrt(dx**2 + dy**2)
    r050 = int(np.sum(dist < 0.5))
    r075 = int(np.sum(dist < 0.75))
    print(f'<0.5: {r050} ({100*r050/n_in:.0f}%)')
    print(f'<0.75: {r075} ({100*r075/n_in:.0f}%)')

    # 检查投影到图像外的星的分布
    out_img = ~in_img
    n_out = int(np.sum(out_img))
    print(f'Out of image: {n_out}')
    if n_out > 0:
        out_px = px_arr[out_img]
        out_py = py_arr[out_img]
        print(f'  Out px range: [{out_px.min():.0f}, {out_px.max():.0f}]')
        print(f'  Out py range: [{out_py.min():.0f}, {out_py.max():.0f}]')

    client.close()
