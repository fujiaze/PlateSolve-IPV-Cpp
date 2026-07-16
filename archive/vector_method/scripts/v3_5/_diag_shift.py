"""诊断平移残差: Umeyama vs CD+CRVAL vs 检测星坐标"""
import sys,os,numpy as np,math,json
sys.path.insert(0,'lib/plate_solve/python')
sys.path.insert(0,'lib/astro_image_io/python')
sys.path.insert(0,'lib/star_detector/python')
from astro_image_io import ImageReader
from vector_match_v3_5_cpp import VectorMatchV35Cpp
from vector_match_v2 import GaiaClientPy, _build_catalog_vectors, _apply_flip
from star_detector import StarDetector, SDetParamsPy
from astropy.io import fits as afits
from astropy.coordinates import SkyCoord
import astropy.units as u
from scipy.spatial import cKDTree

os.chdir(r"F:\Astro dev\Astro CS Normalization Database")

for fits_path, label in [
    ("testdata/lights/NGC7293_T2_HO_flying_dutchman-20250607@085204-1200S-H-alpha.fts","NGC7293"),
    ("testdata/lights1/panel1/Galaxy_Center_mosaic1_T4_flying_dutchman-20250813@011752-180S-Red.fts","GC_P1"),
]:
    print(f"\n{'='*60} {label} {'='*60}")
    reader=ImageReader(); img=reader.read(fits_path)
    w,h=img.width,img.height; fl=img.metadata.observation.focallen
    ps=img.metadata.observation.xpixsz; s0=206.265*ps/fl
    hdul=afits.open(fits_path); hdr=hdul[0].header; hdul.close()
    exptime=float(hdr.get('EXPTIME',1.0))
    sc=SkyCoord(hdr.get('RA',hdr.get('OBJCTRA','')),hdr.get('DEC',hdr.get('OBJCTDEC','')),
                unit=(u.hourangle,u.deg))
    cra,cdec=sc.ra.deg,sc.dec.deg; cos_d=math.cos(cdec*math.pi/180.0)

    detector=StarDetector(params=SDetParamsPy(fitRadius=0))
    det=detector.detect_ex(img.data)

    vm=VectorMatchV35Cpp('GaiaDR3')
    result=vm.solve(np.array(det.x,np.float64),np.array(det.y,np.float64),
                     np.array(det.flux,np.float64),np.array(det.saturated,np.int32),
                     cra,cdec,fl,ps,w,h,wcs_out='overlay_output/_diag.json',
                     skip_sip=False,exptime=exptime)
    vm.close()
    if not result: print("FAIL"); continue

    with open('overlay_output/_diag.json') as f: wcs=json.load(f)
    cd=np.array(wcs['CD'],np.float64); crval=np.array(wcs['CRVAL'],np.float64)
    crpix=np.array(wcs['CRPIX'],np.float64)
    sip_A=np.array(wcs.get('SIP_A',[0]*36),np.float64).reshape(6,6)
    sip_B=np.array(wcs.get('SIP_B',[0]*36),np.float64).reshape(6,6)
    sip_o=wcs.get('SIP_ORDER',0)

    s_s=result.solve_s; th=math.radians(result.rotation_deg)
    fm=result.flip_mode; fx=(fm==1 or fm==3); fy=(fm==2 or fm==3)
    ct,st=math.cos(th),math.sin(th)

    print(f"s={s_s:.4f} θ={result.rotation_deg:.2f}° tx={result.solve_tx:.2f}\" ty={result.solve_ty:.2f}\" flip={fm} s0={s0:.4f}\"/px")
    print(f"CRVAL={crval[0]:.10f} {crval[1]:.10f} CRPIX={crpix}")
    print(f"SIP_RMS={wcs['RMS_PX']:.4f}px order={sip_o}")

    # 查询全部Gaia星
    fov_d=math.sqrt(w*w+h*h)*s0/3600.0
    gaia=GaiaClientPy('GaiaDR3',1)
    ra_a,dec_a,_=gaia.cone_search(cra,cdec,fov_d*0.55,22.0)
    gaia.close()
    ra_a=np.array(ra_a,np.float64); dec_a=np.array(dec_a,np.float64)

    # 构建Wf
    W=_build_catalog_vectors(ra_a,dec_a,cra,cdec)
    Wf=_apply_flip(W,fm)

    # === A: Umeyama直接投影 ===
    xA=np.full(len(ra_a),np.nan); yA=np.full(len(ra_a),np.nan)
    for i in range(len(ra_a)):
        wx,wy=Wf[i,0],Wf[i,1]
        ux=s_s*(ct*wx-st*wy)+result.solve_tx
        uy=s_s*(st*wx+ct*wy)+result.solve_ty
        xA[i]=ux/s0+w/2.0; yA[i]=-uy/s0+h/2.0

    # === B: CD+CRVAL线性 ===
    cdet=cd[0,0]*cd[1,1]-cd[0,1]*cd[1,0]
    cdi=np.array([[cd[1,1],-cd[0,1]],[-cd[1,0],cd[0,0]]])/cdet
    xi_p=cdi[0,0]*(ra_a-crval[0])+cdi[0,1]*(dec_a-crval[1])
    et_p=cdi[1,0]*(ra_a-crval[0])+cdi[1,1]*(dec_a-crval[1])
    xB=xi_p+crpix[0]; yB=et_p+crpix[1]

    # === C: CD+CRVAL+SIP ===
    ifv=(xB>-500)&(xB<w+500)&(yB>-500)&(yB<h+500)
    xi_s=xi_p[ifv].copy(); et_s=et_p[ifv].copy()
    xi_o=xi_p[ifv].copy(); et_o=et_p[ifv].copy()
    mo=min(sip_o,6) if sip_o>0 else 0
    sts=[]
    if mo>=2:
        for p in range(mo+1):
            for q in range(mo+1):
                if p+q<2 or p+q>mo: continue
                if p>=6 or q>=6: continue
                if abs(sip_A[p,q])>1e-30 or abs(sip_B[p,q])>1e-30:
                    sts.append((p,q,sip_A[p,q],sip_B[p,q]))
    for _ in range(20):
        sdx=np.zeros_like(xi_s); sdy=np.zeros_like(et_s)
        for p,q,ac,bc in sts:
            xc=np.clip(xi_s,-1e4,1e4); yc=np.clip(et_s,-1e4,1e4)
            tm=xc**p*yc**q; tm=np.where(np.isfinite(tm),tm,0)
            sdx+=ac*tm; sdy+=bc*tm
        xn=xi_o-sdx; yn=et_o-sdy
        if np.max(np.abs(xn-xi_s))<1e-6 and np.max(np.abs(yn-et_s))<1e-6: break
        xi_s, et_s = xn, yn
    xC=np.full(len(ra_a),np.nan); yC=np.full(len(ra_a),np.nan)
    xC[ifv]=xi_s+crpix[0]; yC[ifv]=et_s+crpix[1]

    # 对比
    iA=np.isfinite(xA)&(xA>0)&(xA<w)&(yA>0)&(yA<h)
    iB=(xB>0)&(xB<w)&(yB>0)&(yB<h)
    iC=np.isfinite(xC)&(xC>0)&(xC<w)&(yC>0)&(yC<h)

    cm=iA&iB
    if cm.sum()>3:
        print(f"\nUmeyama→CD ({cm.sum()}星): dx={np.mean(xA[cm]-xB[cm]):+.3f}±{np.std(xA[cm]-xB[cm]):.3f}px  dy={np.mean(yA[cm]-yB[cm]):+.3f}±{np.std(yA[cm]-yB[cm]):.3f}px")

    cm2=iB&iC
    if cm2.sum()>3:
        print(f"CD→SIP ({cm2.sum()}星): dx={np.mean(xB[cm2]-xC[cm2]):+.3f}±{np.std(xB[cm2]-xC[cm2]):.3f}px  dy={np.mean(yB[cm2]-yC[cm2]):+.3f}±{np.std(yB[cm2]-yC[cm2]):.3f}px")

    # 对比检测星
    dx=np.array(det.x,np.float64); dy=np.array(det.y,np.float64)
    td=cKDTree(np.column_stack([dx,dy]))

    for name,xx,yy,ii in [("A:Umeyama",xA,yA,iA),("B:CD lin",xB,yB,iB),("C:CD+SIP",xC,yC,iC)]:
        if ii.sum()<3: continue
        gp=np.column_stack([xx[ii],yy[ii]])
        ds,_=td.query(gp,k=1)
        ok=ds<max(2,max(2,2*s0))
        if ok.sum()<3: continue
        rx=gp[ok,0]-dx[_[ok]]; ry=gp[ok,1]-dy[_[ok]]
        print(f"{name}→检测星({ok.sum()}对): dx={np.mean(rx):+.3f}±{np.std(rx):.3f}px med={np.median(rx):+.3f}  dy={np.mean(ry):+.3f}±{np.std(ry):.3f}px med={np.median(ry):+.3f}")
