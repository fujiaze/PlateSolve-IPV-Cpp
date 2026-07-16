"""
用真实Phase D输出验证: 从C++提取clean pairs→独立拟合→对比C++结果
"""
import sys,os,numpy as np,math,json
sys.path.insert(0,'lib/plate_solve/python')
sys.path.insert(0,'lib/astro_image_io/python')
sys.path.insert(0,'lib/star_detector/python')
from astro_image_io import ImageReader
from vector_match_v3_5_cpp import VectorMatchV35Cpp
from vector_match_v2 import _build_catalog_vectors, _apply_flip
from star_detector import StarDetector, SDetParamsPy
from astropy.io import fits as afits
from astropy.coordinates import SkyCoord
import astropy.units as u
from test_fit_standalone import fit_pipeline, generate_synthetic, umeyama

os.chdir(r"F:\Astro dev\Astro CS Normalization Database")

def run_real_test(fits_path, label):
    print(f"\n{'='*60}")
    print(f"REAL: {label}")
    print(f"{'='*60}")
    reader=ImageReader(); img=reader.read(fits_path)
    w,h=img.width,img.height; fl=img.metadata.observation.focallen
    ps=img.metadata.observation.xpixsz; s0=206.265*ps/fl
    hdul=afits.open(fits_path); hdr=hdul[0].header; hdul.close()
    exptime=float(hdr.get('EXPTIME',1.0))
    sc=SkyCoord(hdr.get('RA',''),hdr.get('DEC',''),unit=(u.hourangle,u.deg))
    cra0,cdec0=sc.ra.deg,sc.dec.deg; cd0=math.cos(cdec0*math.pi/180)

    detector=StarDetector(params=SDetParamsPy(fitRadius=0))
    det=detector.detect_ex(img.data)
    dx=np.array(det.x,np.float64); dy=np.array(det.y,np.float64)

    # 运行C++ solve (仅提取Phase D数据)
    vm=VectorMatchV35Cpp('GaiaDR3')
    result=vm.solve(dx,dy,np.array(det.flux,np.float64),np.array(det.saturated,np.int32),
                     cra0,cdec0,fl,ps,w,h,
                     wcs_out='overlay_output/_real_test.json',
                     skip_sip=False,exptime=exptime)
    vm.close()
    if not result:
        print("  FAILED")
        return

    with open('overlay_output/_real_test.json') as f: wcs=json.load(f)
    print(f"C++输出: s={result.solve_s:.4f} θ={result.rotation_deg:.1f}° "
          f"SIP_RMS={wcs['RMS_PX']:.3f}px order={wcs.get('SIP_ORDER',0)}")

    # 重建Phase D clean pairs — 用1对1互斥匹配
    from scipy.spatial import cKDTree
    gaia_v2 = __import__('vector_match_v2').GaiaClientPy
    gaia_cpp = gaia_v2('GaiaDR3',1)
    fov_d = math.sqrt(w*w+h*h)*s0/3600.0
    ra_a,dec_a,_=gaia_cpp.cone_search(cra0,cdec0,fov_d*0.55,22.0)
    gaia_cpp.close()
    ra_a=np.array(ra_a,np.float64); dec_a=np.array(dec_a,np.float64)

    W=_build_catalog_vectors(ra_a,dec_a,cra0,cdec0)
    fm=result.flip_mode; fx=(fm==1 or fm==3); fy=(fm==2 or fm==3)
    Wf=_apply_flip(W,fm)

    s_s=result.solve_s; th=math.radians(result.rotation_deg)
    ct,st=math.cos(th),math.sin(th)

    # Umeyama投影→1对1匹配
    gx_u=np.zeros(len(ra_a)); gy_u=np.zeros(len(ra_a))
    for i in range(len(ra_a)):
        ux=s_s*(ct*Wf[i,0]-st*Wf[i,1])+result.solve_tx
        uy=s_s*(st*Wf[i,0]+ct*Wf[i,1])+result.solve_ty
        gx_u[i]=ux/s0+w/2.; gy_u[i]=-uy/s0+h/2.

    td=cKDTree(np.column_stack([dx,dy]))
    g_ok=np.isfinite(gx_u)&(gx_u>0)&(gx_u<w)&(gy_u>0)&(gy_u<h)
    gp=np.column_stack([gx_u[g_ok],gy_u[g_ok]]); gi_list=np.where(g_ok)[0]
    ds,ids=td.query(gp,k=1)
    used=np.zeros(len(dx),dtype=bool)
    pairs=[]
    for kk in np.argsort(ds):
        if ds[kk]>2:
            break
        ii=ids[kk]
        if used[ii]:
            continue
        used[ii]=True
        pairs.append((ii,gi_list[kk]))

    npairs=len(pairs)
    print(f"1对1匹配<2px: {npairs}对")

    if npairs<10: return

    # 构建独立拟合输入 (完全复刻C++格式)
    U_data = np.zeros((npairs, 2))
    Wf_data = np.zeros((npairs, 2))
    for k,(di,gi) in enumerate(pairs):
        U_data[k,0] = (dx[di] - w/2.) * s0
        U_data[k,1] = -(dy[di] - h/2.) * s0
        Wf_data[k,0] = Wf[gi,0]
        Wf_data[k,1] = Wf[gi,1]

    synth = {
        'gt': {
            'CD': np.eye(2), 'CRVAL': np.array([cra0, cdec0]),
            'CRPIX': np.array([w/2., h/2.]), 's0': s0, 'w': w, 'h': h,
            'SIP_A': np.zeros((6,6)), 'SIP_B': np.zeros((6,6)), 'sip_order': 0,
            'center_ra': cra0, 'center_dec': cdec0, 'cos_dec': cd0,
            's_true': s_s, 'theta_deg': result.rotation_deg, 'flip_mode': fm,
            'fx': fx, 'fy': fy, 'sx': -1.0 if fx else 1.0, 'sy': -1.0 if fy else 1.0,
        },
        'pairs': {
            'U': U_data, 'Wf': Wf_data,
            'U_pix': np.column_stack([dx[[p[0] for p in pairs]], dy[[p[0] for p in pairs]]]),
            'gaia_pix': np.column_stack([gx_u[[p[1] for p in pairs]], gy_u[[p[1] for p in pairs]]]),
            'gaia_ra': ra_a[[p[1] for p in pairs]], 'gaia_dec': dec_a[[p[1] for p in pairs]],
        },
        'noise_sigma': 0,
    }

    r = fit_pipeline(synth, sip_max_order=4)
    print(f"Python管线: 仿射RMS={r['affine_RMS']:.4f}px  SIP阶={r['sip_order']} "
          f"SIP_RMS={r['sip_RMS']:.4f}px  最终RMS={r['final_RMS']:.4f}px")
    print(f"  CD={r['CD'][0,0]:.6e},{r['CD'][0,1]:.6e}; "
          f"{r['CD'][1,0]:.6e},{r['CD'][1,1]:.6e}")
    print(f"  s={r['s']:.4f} θ={r['theta_deg']:.2f}°")

    # 对比C++ vs Python
    cd_cpp = np.array(wcs['CD'])
    print(f"  C++ CD={cd_cpp[0,0]:.6e},{cd_cpp[0,1]:.6e}; "
          f"{cd_cpp[1,0]:.6e},{cd_cpp[1,1]:.6e}")
    cd_diff = np.max(np.abs(r['CD']-cd_cpp))
    crval_cpp = np.array(wcs['CRVAL'])
    crval_diff = np.max(np.abs(r['CRVAL']-crval_cpp))*3600
    print(f"  CD偏差={cd_diff:.2e}  CRVAL偏差={crval_diff:.2f}\"")
    print(f"  C++ SIP_RMS={wcs['RMS_PX']:.4f}px  Python SIP_RMS={r['sip_RMS']:.4f}px")

    # 用真实匹配对和ground truth CD重建后计算理论残差
    # 比较 Umeyama直投 vs Python管线
    ux_all=np.zeros(npairs); uy_all=np.zeros(npairs)
    for k,(di,gi) in enumerate(pairs):
        ux_all[k]=s_s*(ct*Wf_data[k,0]-st*Wf_data[k,1])+result.solve_tx
        uy_all[k]=s_s*(st*Wf_data[k,0]+ct*Wf_data[k,1])+result.solve_ty
    px_all = ux_all/s0+w/2.; py_all = -uy_all/s0+h/2.
    dx_pair = dx[[p[0] for p in pairs]]; dy_pair = dy[[p[0] for p in pairs]]
    res_um = np.column_stack([px_all-dx_pair, py_all-dy_pair])
    rms_um = math.sqrt(np.mean(res_um[:,0]**2+res_um[:,1]**2))
    print(f"  Umeyama直投RMS(配对后)={rms_um:.4f}px")

run_real_test("testdata/lights/NGC7293_T2_HO_flying_dutchman-20250607@085204-1200S-H-alpha.fts","NGC7293_Ha")
run_real_test("testdata/lights1/panel2/Galaxy_Center_mosaic2_T4_flying_dutchman-20250717@022723-600S-Oiii.fts","GC_P2_Oiii")
