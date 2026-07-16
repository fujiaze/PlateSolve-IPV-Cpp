"""用Phase D clean对直接验证WCS投影精度: src vs CD+SIP投影dst"""
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
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

os.chdir(r"F:\Astro dev\Astro CS Normalization Database")
OUT = "overlay_output"

def diagnose_phaseD_residuals(fits_path, label):
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

    wcs_out=os.path.join(OUT,f'_clean_{label}.json')
    vm=VectorMatchV35Cpp('GaiaDR3')
    result=vm.solve(np.array(det.x,np.float64),np.array(det.y,np.float64),
                     np.array(det.flux,np.float64),np.array(det.saturated,np.int32),
                     cra0,cdec0,fl,ps,w,h,wcs_out=wcs_out,skip_sip=False,exptime=exptime)
    vm.close()
    if not result: print(f"{label}: SOLVE FAILED"); return

    with open(wcs_out) as f: wcs=json.load(f)
    cd=np.array(wcs['CD'],np.float64); crval=np.array(wcs['CRVAL'],np.float64)
    crpix=np.array(wcs['CRPIX'],np.float64)
    sip_A=np.array(wcs.get('SIP_A',[0]*36),np.float64).reshape(6,6)
    sip_B=np.array(wcs.get('SIP_B',[0]*36),np.float64).reshape(6,6)
    sip_o=wcs.get('SIP_ORDER',0)

    s_s=result.solve_s; th=math.radians(result.rotation_deg)
    fm=result.flip_mode; fx=(fm==1 or fm==3); fy=(fm==2 or fm==3)
    ct,st=math.cos(th),math.sin(th)

    print(f"\n{'='*70}")
    print(f"{label}: s={s_s:.6f} θ={result.rotation_deg:.3f}° flip={fm} s0={s0:.4f}\"/px")
    print(f"CD={cd[0]:.6e},{cd[1]:.6e}  CRVAL={crval[0]:.10f},{crval[1]:.10f}  CRPIX={crpix}")
    print(f"SIP order={sip_o} RMS={wcs['RMS_PX']:.4f}px")
    print(f"tx={result.solve_tx:.2f}\" ty={result.solve_ty:.2f}\"")

    # 用s/θ/fx/fy重建Phase D的对应关系
    # gnomonic投影→flip→准确对应
    # 我们需要重建: 哪些Gaia星被Phase D选中了
    # 方法: 对全部Gaia跑Umeyama直投，与检测星做1对1互斥匹配
    
    import ctypes
    from vector_match_v2 import GaiaClientPy as GC
    gaia_cpp=GC('GaiaDR3',1)
    fov_d=math.sqrt(w*w+h*h)*s0/3600.0
    ra_a,dec_a,mag_a=gaia_cpp.cone_search(cra0,cdec0,fov_d*0.55,22.0)
    gaia_cpp.close()
    ra_a=np.array(ra_a,np.float64); dec_a=np.array(dec_a,np.float64)

    # 构建arcsec域坐标
    W=_build_catalog_vectors(ra_a,dec_a,cra0,cdec0)
    Wf=_apply_flip(W,fm)

    # Umeyama直接投影→像素 (与C++ Phase E一致)
    U_all=np.column_stack([(dx-crpix[0])*s0, -(dy-crpix[1])*s0])

    # 1对1互斥匹配: Wf→Umeyama→像素 vs 检测星
    wx_all,wy_all=Wf[:,0],Wf[:,1]
    gx_all=np.full(len(ra_a),np.nan); gy_all=np.full(len(ra_a),np.nan)
    for i in range(len(ra_a)):
        ux=s_s*(ct*wx_all[i]-st*wy_all[i])+result.solve_tx
        uy=s_s*(st*wx_all[i]+ct*wy_all[i])+result.solve_ty
        gx_all[i]=ux/s0+crpix[0]; gy_all[i]=-uy/s0+crpix[1]

    # KDTree匹配 + 1对1互斥
    from scipy.spatial import cKDTree
    td=cKDTree(np.column_stack([dx,dy]))
    g_ok = np.isfinite(gx_all)&(gx_all>0)&(gx_all<w)&(gy_all>0)&(gy_all<h)
    g_pts = np.column_stack([gx_all[g_ok],gy_all[g_ok]])
    g_idx = np.where(g_ok)[0]
    ds_all,ids_all=td.query(g_pts,k=1)
    # 1对1互斥(距离<3px)
    used_d=np.zeros(len(dx),dtype=bool)
    clean_pairs=[]
    order=np.argsort(ds_all)
    for kk in order:
        if ds_all[kk]>3: continue
        ii=ids_all[kk]
        if used_d[ii]: continue
        used_d[ii]=True
        clean_pairs.append((ii, g_idx[kk]))

    ncp=len(clean_pairs)
    print(f"1对1匹配(3px): {ncp}对 (总Gaia: {len(ra_a)}星, FOV内: {g_ok.sum()}星)")

    if ncp<10:
        print("匹配对不足,跳过")
        return

    # WCS-SIP投影Gaia
    cdet=cd[0,0]*cd[1,1]-cd[0,1]*cd[1,0]
    cdi=np.array([[cd[1,1],-cd[0,1]],[-cd[1,0],cd[0,0]]])/cdet
    xi_p=cdi[0,0]*(ra_a-crval[0])+cdi[0,1]*(dec_a-crval[1])
    et_p=cdi[1,0]*(ra_a-crval[0])+cdi[1,1]*(dec_a-crval[1])
    x_lin=xi_p+crpix[0]; y_lin=et_p+crpix[1]
    mrg=500; ifv=(x_lin>-mrg)&(x_lin<w+mrg)&(y_lin>-mrg)&(y_lin<h+mrg)
    xi=xi_p[ifv].copy(); et=et_p[ifv].copy()
    xio=xi_p[ifv].copy(); eto=et_p[ifv].copy()
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
        sdx=np.zeros_like(xi); sdy=np.zeros_like(et)
        for p,q,ac,bc in sts:
            xc=np.clip(xi,-1e4,1e4); yc=np.clip(et,-1e4,1e4)
            tm=xc**p*yc**q; tm=np.where(np.isfinite(tm),tm,0)
            sdx+=ac*tm; sdy+=bc*tm
        xn=xio-sdx; yn=eto-sdy
        if np.max(np.abs(xn-xi))<1e-6 and np.max(np.abs(yn-et))<1e-6: break
        xi,et=xn,yn
    x_sip=np.full(len(ra_a),np.nan); y_sip=np.full(len(ra_a),np.nan)
    x_sip[ifv]=xi+crpix[0]; y_sip[ifv]=et+crpix[1]

    # 对每个clean pair计算3种投影vs检测星残差
    rx_um=[]; ry_um=[]; rx_cd=[]; ry_cd=[]; rx_sip=[]; ry_sip=[]
    for (di,gi) in clean_pairs:
        rx_um.append(dx[di]-gx_all[gi])
        ry_um.append(dy[di]-gy_all[gi])
        if np.isfinite(x_sip[gi]):
            rx_cd.append(dx[di]-x_lin[gi])
            ry_cd.append(dy[di]-y_lin[gi])
            rx_sip.append(dx[di]-x_sip[gi])
            ry_sip.append(dy[di]-y_sip[gi])

    for name,rxl,ryl in [("Umeyama直投",rx_um,ry_um),("CD线性",rx_cd,ry_cd),("CD+SIP",rx_sip,ry_sip)]:
        if not rxl: continue
        rxa=np.array(rxl); rya=np.array(ryl)
        print(f"\n{name} ({len(rxa)}对):")
        print(f"  med=[{np.median(rxa):+.3f},{np.median(rya):+.3f}]px "
              f"mean=[{np.mean(rxa):+.3f},{np.mean(rya):+.3f}]px "
              f"std=[{np.std(rxa):.3f},{np.std(rya):.3f}]px "
              f"RMS={np.sqrt(np.mean(rxa**2+rya**2)):.3f}px")

    # 9区域分析(CD+SIP)
    if len(rx_sip)<20: return
    rxa=np.array(rx_sip); rya=np.array(ry_sip)
    bw,bh=w/3,h/3
    row_names=["上左","上中","上右","中左","中中","中右","下左","下中","下右"]
    print("\n--- CD+SIP 9区域(仅Phase D匹配对) ---")
    for ri in range(9):
        rj=ri//3; ci=ri%3
        m=np.zeros(len(rxa),dtype=bool)
        for k,(di,_) in enumerate(clean_pairs):
            rr=int(dy[di]//bh); cc=int(dx[di]//bw)
            if rr==rj and cc==ci: m[k]=True
        if m.sum()<2:
            print(f"  [{row_names[ri]}] n<2")
            continue
        mx=np.median(rxa[m]); my=np.median(rya[m])
        ax=np.mean(rxa[m]); ay=np.mean(rya[m])
        ang=math.degrees(math.atan2(my,mx))
        mag=math.sqrt(mx*mx+my*my)
        print(f"  [{row_names[ri]}] n={int(m.sum()):3d}  med=[{mx:+.3f},{my:+.3f}]px  "
              f"方向={ang:+.1f}° 幅={mag:.3f}px  mean=[{ax:+.3f},{ay:+.3f}]px")

    # 渲染残差向量图
    data=img.data.astype(np.float32)
    dd=data[data>0]; lo,hi=np.percentile(dd,(1,99.5)) if len(dd)>1 else (0,1)
    ims=np.clip((data-lo)/max(hi-lo,1),0,1)
    DPI=100
    fig=plt.figure(figsize=(w/DPI,h/DPI),dpi=DPI,frameon=False)
    ax=fig.add_axes([0,0,1,1])
    ax.imshow(ims,cmap="gray",origin="lower",interpolation="nearest")
    for i in range(1,3):
        ax.axvline(i*bw,color='cyan',lw=0.5,alpha=0.3)
        ax.axhline(i*bh,color='cyan',lw=0.5,alpha=0.3)

    # 每个clean pair画残差箭头(红色: 检测星→Gaia投影)
    for k,(di,gi) in enumerate(clean_pairs):
        if k>=len(rx_sip): continue
        ax.arrow(dx[di],dy[di],-rx_sip[k],-ry_sip[k],  # -残差=Gaia→检测
                 head_width=10,head_length=6,fc='red',ec='red',alpha=0.7,lw=0.2,
                 length_includes_head=True)

    # 各区域均值箭头
    colors=['yellow','cyan','magenta','lime','orange','pink','white','#0ff','#f0f']
    for ri in range(9):
        rj=ri//3; ci=ri%3
        m=np.zeros(len(rxa),dtype=bool)
        for k,(di,_) in enumerate(clean_pairs):
            rr=int(dy[di]//bh); cc=int(dx[di]//bw)
            if rr==rj and cc==ci: m[k]=True
        if m.sum()<3: continue
        mx=np.mean(rxa[m]); my=np.mean(rya[m])
        cx_r=ci*bw+bw/2; cy_r=rj*bh+bh/2
        sc=15
        ax.arrow(cx_r,cy_r,mx*sc,my*sc,head_width=30,head_length=20,
                 fc=colors[ri],ec=colors[ri],alpha=0.9,lw=2,length_includes_head=True)
        ax.text(cx_r,cy_r-35,f'{mx*sc:+.1f},{my*sc:+.1f}',color=colors[ri],fontsize=8,ha='center',
                bbox=dict(boxstyle='round',facecolor='black',alpha=0.6))

    ax.set_xlim(0,w); ax.set_ylim(0,h); ax.axis("off")
    out=f'{OUT}/_clean_residual_{label}.png'
    fig.savefig(out,dpi=DPI,pad_inches=0)
    plt.close(fig)
    print(f"残差图: {out}")

for fits_path,label in [
    ("testdata/lights/NGC7293_T2_HO_flying_dutchman-20250607@085204-1200S-H-alpha.fts","NGC7293_Ha"),
    ("testdata/lights1/panel2/Galaxy_Center_mosaic2_T4_flying_dutchman-20250717@022723-600S-Oiii.fts","GC_P2_Oiii"),
    ("testdata/lights1/panel1/Galaxy_Center_mosaic1_T4_flying_dutchman-20250813@011752-180S-Red.fts","GC_P1_Red"),
]:
    diagnose_phaseD_residuals(fits_path,label)
