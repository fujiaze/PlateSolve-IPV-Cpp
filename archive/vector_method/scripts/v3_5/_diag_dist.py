"""诊断真实匹配对残差分布 + CD投影正确性"""
import sys,os,numpy as np,math,json
sys.path.insert(0,'lib/plate_solve/python')
sys.path.insert(0,'lib/astro_image_io/python')
sys.path.insert(0,'lib/star_detector/python')
from astro_image_io import ImageReader
from vector_match_v3_5_cpp import VectorMatchV35Cpp
from vector_match_v2 import GaiaClientPy,_build_catalog_vectors,_apply_flip
from star_detector import StarDetector,SDetParamsPy
from astropy.io import fits as afits
from astropy.coordinates import SkyCoord
import astropy.units as u
from scipy.spatial import cKDTree

os.chdir(r"F:\Astro dev\Astro CS Normalization Database")

for fits_path,label in [
    ("testdata/lights/NGC7293_T2_HO_flying_dutchman-20250607@085204-1200S-H-alpha.fts","NGC7293"),
    ("testdata/lights1/panel2/Galaxy_Center_mosaic2_T4_flying_dutchman-20250717@022723-600S-Oiii.fts","GC_P2"),
]:
    print(f"\n{'='*60}")
    print(f"{label}")
    reader=ImageReader();img=reader.read(fits_path)
    w,h=img.width,img.height;fl=img.metadata.observation.focallen
    ps=img.metadata.observation.xpixsz;s0=206.265*ps/fl
    hdul=afits.open(fits_path);hdr=hdul[0].header;hdul.close()
    exptime=float(hdr.get('EXPTIME',1.0))
    sc=SkyCoord(hdr.get('RA',''),hdr.get('DEC',''),unit=(u.hourangle,u.deg))
    cra,cdec=sc.ra.deg,sc.dec.deg;cd0=math.cos(cdec*math.pi/180)
    detector=StarDetector(params=SDetParamsPy(fitRadius=0))
    det=detector.detect_ex(img.data)
    dx=np.array(det.x,np.float64);dy=np.array(det.y,np.float64)

    vm=VectorMatchV35Cpp('GaiaDR3')
    result=vm.solve(dx,dy,np.array(det.flux,np.float64),np.array(det.saturated,np.int32),
                     cra,cdec,fl,ps,w,h,wcs_out='overlay_output/_diag_q.json',
                     skip_sip=False,exptime=exptime)
    vm.close()
    if not result:print("FAIL");continue
    with open('overlay_output/_diag_q.json') as f:wc=json.load(f)

    fm=result.flip_mode;fx=(fm==1 or fm==3);fy=(fm==2 or fm==3)
    sx=-1 if fx else 1;sy=-1 if fy else 1
    s_s=result.solve_s;th=math.radians(result.rotation_deg)
    ct,st=math.cos(th),math.sin(th)

    # 构建Wf → 1对1匹配
    gaia=GaiaClientPy('GaiaDR3',1)
    fov_d=math.sqrt(w*w+h*h)*s0/3600.
    ra_a,dec_a,_=gaia.cone_search(cra,cdec,fov_d*0.55,22.0);gaia.close()
    ra_a=np.array(ra_a,np.float64);dec_a=np.array(dec_a,np.float64)
    W=_build_catalog_vectors(ra_a,dec_a,cra,cdec);Wf=_apply_flip(W,fm)

    gx=np.zeros(len(ra_a));gy=np.zeros(len(ra_a))
    for i in range(len(ra_a)):
        ux=s_s*(ct*Wf[i,0]-st*Wf[i,1])+result.solve_tx
        uy=s_s*(st*Wf[i,0]+ct*Wf[i,1])+result.solve_ty
        gx[i]=ux/s0+w/2.;gy[i]=-uy/s0+h/2.
    td=cKDTree(np.column_stack([dx,dy]))
    g_ok=np.isfinite(gx)&(gx>0)&(gx<w)&(gy>0)&(gy<h)
    gp=np.column_stack([gx[g_ok],gy[g_ok]]);gil=np.where(g_ok)[0]
    ds,ids=td.query(gp,k=1)
    used=np.zeros(len(dx),bool);pairs=[]
    for kk in np.argsort(ds):
        if ds[kk]>5:
            break
        ii=ids[kk]
        if used[ii]:
            continue
        used[ii]=True
        pairs.append((ii,gil[kk]))
    npairs=len(pairs)
    print(f"1对1匹配<5px: {npairs}对")

    # === 验证CD投影 ===
    cd_c=np.array(wc['CD'],np.float64);crv=np.array(wc['CRVAL'],np.float64)
    crp=np.array(wc['CRPIX'],np.float64)
    cdet=cd_c[0,0]*cd_c[1,1]-cd_c[0,1]*cd_c[1,0]
    cdi=np.array([[cd_c[1,1],-cd_c[0,1]],[-cd_c[1,0],cd_c[0,0]]])/cdet

    res_um=[];res_cd=[];res_wf=[]
    for di,gi in pairs:
        # Umeyama直投
        ux=s_s*(ct*Wf[gi,0]-st*Wf[gi,1])+result.solve_tx
        uy=s_s*(st*Wf[gi,0]+ct*Wf[gi,1])+result.solve_ty
        px_um=ux/s0+w/2.;py_um=-uy/s0+h/2.
        res_um.append((dx[di]-px_um,dy[di]-py_um))

        # CD投影
        Wx=-Wf[gi,0] if fx else Wf[gi,0]
        Wy=-Wf[gi,1] if fy else Wf[gi,1]
        dra_deg=Wx/(3600.*cd0);ddec_deg=Wy/3600.
        xi_cd=cdi[0,0]*dra_deg+cdi[0,1]*ddec_deg
        eta_cd=cdi[1,0]*dra_deg+cdi[1,1]*ddec_deg
        px_cd=xi_cd+crp[0];py_cd=eta_cd+crp[1]
        res_cd.append((dx[di]-px_cd,dy[di]-py_cd))

        # Wf直接用
        res_wf.append((Wf[gi,0],Wf[gi,1]))

    # 残差分布
    r_um=np.array(res_um);r_cd=np.array(res_cd)
    for name,rr in [("Umeyama直投",r_um),("CD投影",r_cd)]:
        rx,ry=rr[:,0],rr[:,1]
        rdist=np.sqrt(rx**2+ry**2)
        print(f"\n{name}:")
        print(f"  RMS={np.sqrt(np.mean(rx**2+ry**2)):.3f}px")
        print(f"  med=[{np.median(rx):+.3f},{np.median(ry):+.3f}]px")
        print(f"  P10={np.percentile(rdist,10):.2f} P50={np.percentile(rdist,50):.2f} "
              f"P90={np.percentile(rdist,90):.2f} P95={np.percentile(rdist,95):.2f} "
              f"max={rdist.max():.2f}")
        # 直方图
        bins=[0,0.5,1,2,3,5,10,20,50,100,999]
        for i in range(len(bins)-1):
            cnt=((rdist>=bins[i])&(rdist<bins[i+1])).sum()
            if cnt>0:print(f"  [{bins[i]:5.1f},{bins[i+1]:5.1f})px: {cnt:4d}")

    # 算Umeyama用自己的s/θ做初值
    us0=np.column_stack([(dx[[p[0] for p in pairs]]-w/2.)*s0,
                          -(dy[[p[0] for p in pairs]]-h/2.)*s0])
    ws0=np.column_stack([Wf[[p[1] for p in pairs],0],
                         Wf[[p[1] for p in pairs],1]])
    ms=np.mean(ws0,axis=0);md=np.mean(us0,axis=0)
    sc_=ws0-ms;dc_=us0-md
    H=sc_.T@dc_;U_,S_,Vt_=np.linalg.svd(H)
    det_=np.linalg.det(Vt_.T@U_.T);Sv_=np.array([1,det_])
    R_=Vt_.T@np.diag(Sv_)@U_.T
    s_um=np.dot(S_,Sv_)/np.sum(sc_**2)
    th_um=math.atan2(R_[1,0],R_[0,0])
    tx_um=md[0]-s_um*(math.cos(th_um)*ms[0]-math.sin(th_um)*ms[1])
    ty_um=md[1]-s_um*(math.sin(th_um)*ms[0]+math.cos(th_um)*ms[1])
    print(f"\nPython Umeyama: s={s_um:.6f} θ={math.degrees(th_um):.2f}° "
          f"tx={tx_um:.2f}\" ty={ty_um:.2f}\"")
    print(f"C++ Output:     s={s_s:.6f} θ={result.rotation_deg:.2f}° "
          f"tx={result.solve_tx:.2f}\" ty={result.solve_ty:.2f}\"")

    # Python CD
    s3600_um=s0/(s_um*3600.)
    ct_um=math.cos(th_um);st_um=math.sin(th_um)
    cd_py=np.array([[sx*s3600_um*ct_um/cd0,-sx*s3600_um*st_um/cd0],
                     [-sy*s3600_um*st_um,-sy*s3600_um*ct_um]])
    cpdet=cd_py[0,0]*cd_py[1,1]-cd_py[0,1]*cd_py[1,0]
    cpdi=np.array([[cd_py[1,1],-cd_py[0,1]],[-cd_py[1,0],cd_py[0,0]]])/cpdet

    res_pycd=[]
    for di,gi in pairs:
        Wx=-Wf[gi,0] if fx else Wf[gi,0]
        Wy=-Wf[gi,1] if fy else Wf[gi,1]
        dra_deg=Wx/(3600.*cd0);ddec_deg=Wy/3600.
        xi=cpdi[0,0]*dra_deg+cpdi[0,1]*ddec_deg
        eta=cpdi[1,0]*dra_deg+cpdi[1,1]*ddec_deg
        res_pycd.append((dx[di]-(xi+w/2.),dy[di]-(eta+h/2.)))
    rpc=np.array(res_pycd)
    prms=math.sqrt(np.mean(rpc[:,0]**2+rpc[:,1]**2))
    print(f"Python CD投影 RMS={prms:.3f}px "
          f"med=[{np.median(rpc[:,0]):.3f},{np.median(rpc[:,1]):.3f}]px")
    print(f"Python CD: [{cd_py[0,0]:.6e},{cd_py[0,1]:.6e}; "
          f"{cd_py[1,0]:.6e},{cd_py[1,1]:.6e}]")
    print(f"C++    CD: [{cd_c[0,0]:.6e},{cd_c[0,1]:.6e}; "
          f"{cd_c[1,0]:.6e},{cd_c[1,1]:.6e}]")
