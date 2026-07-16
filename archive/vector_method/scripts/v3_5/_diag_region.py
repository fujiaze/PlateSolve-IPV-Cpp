"""
Phase D clean对的WCS残差诊断: CD线性 vs CD+SIP 区域分析
"""
import sys,os,numpy as np,math,json
sys.path.insert(0,'lib/plate_solve/python')
sys.path.insert(0,'lib/astro_image_io/python')
sys.path.insert(0,'lib/star_detector/python')
from astro_image_io import ImageReader
from vector_match_v3_5_cpp import VectorMatchV35Cpp
from vector_match_v2 import GaiaClientPy, _build_catalog_vectors, _apply_flip, gnomonic_forward
from star_detector import StarDetector, SDetParamsPy
from astropy.io import fits as afits
from astropy.coordinates import SkyCoord
import astropy.units as u
from scipy.spatial import cKDTree
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

os.chdir(r"F:\Astro dev\Astro CS Normalization Database")
OUT = "overlay_output"

def run(fits_path, label):
    reader=ImageReader(); img=reader.read(fits_path)
    w,h=img.width,img.height; fl=img.metadata.observation.focallen
    ps=img.metadata.observation.xpixsz; s0=206.265*ps/fl
    hdul=afits.open(fits_path); hdr=hdul[0].header; hdul.close()
    exptime=float(hdr.get('EXPTIME',1.0))
    sc=SkyCoord(hdr.get('RA',''),hdr.get('DEC',''),unit=(u.hourangle,u.deg))
    cra0,cdec0=sc.ra.deg,sc.dec.deg

    detector=StarDetector(params=SDetParamsPy(fitRadius=0))
    det=detector.detect_ex(img.data)
    dx=np.array(det.x,np.float64); dy=np.array(det.y,np.float64)

    wcs_out=os.path.join(OUT,f'_diag2_{label}.json')
    vm=VectorMatchV35Cpp('GaiaDR3')
    result=vm.solve(dx,dy,np.array(det.flux,np.float64),np.array(det.saturated,np.int32),
                     cra0,cdec0,fl,ps,w,h,wcs_out=wcs_out,skip_sip=False,exptime=exptime)
    vm.close()
    if not result: return print(f"{label}: FAIL")

    with open(wcs_out) as f: wcs=json.load(f)
    cd=np.array(wcs['CD'],np.float64); crval=np.array(wcs['CRVAL'],np.float64)
    crpix=np.array(wcs['CRPIX'],np.float64)
    sip_A=np.array(wcs.get('SIP_A',[0]*36),np.float64).reshape(6,6)
    sip_B=np.array(wcs.get('SIP_B',[0]*36),np.float64).reshape(6,6)
    sip_o=wcs.get('SIP_ORDER',0)

    s_s=result.solve_s; th=math.radians(result.rotation_deg)
    fm=result.flip_mode; fx=(fm==1 or fm==3); fy=(fm==2 or fm==3)

    print(f"\n{'='*70}")
    print(f"{label}: s={s_s:.6f} θ={result.rotation_deg:.3f}° flip={fm} s0={s0:.4f}\"/px")
    print(f"SIP order={sip_o} RMS={wcs['RMS_PX']:.4f}px")
    print(f"CD=[{cd[0,0]:.6e},{cd[0,1]:.6e}; {cd[1,0]:.6e},{cd[1,1]:.6e}]")

    # 用1对1互斥匹配找到Phase D级别的干净对应
    gaia=GaiaClientPy('GaiaDR3',1)
    fov_d=math.sqrt(w*w+h*h)*s0/3600.0
    ra_a,dec_a,_=gaia.cone_search(cra0,cdec0,fov_d*0.55,22.0)
    gaia.close()
    ra_a=np.array(ra_a,np.float64); dec_a=np.array(dec_a,np.float64)

    # CD+SIP投影全部Gaia
    cdet=cd[0,0]*cd[1,1]-cd[0,1]*cd[1,0]
    cdi=np.array([[cd[1,1],-cd[0,1]],[-cd[1,0],cd[0,0]]])/cdet
    xi_p=cdi[0,0]*(ra_a-crval[0])+cdi[0,1]*(dec_a-crval[1])
    et_p=cdi[1,0]*(ra_a-crval[0])+cdi[1,1]*(dec_a-crval[1])
    x_cd=xi_p+crpix[0]; y_cd=et_p+crpix[1]

    mrg=500; ifv=(x_cd>-mrg)&(x_cd<w+mrg)&(y_cd>-mrg)&(y_cd<h+mrg)
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

    # 1对1互斥匹配 (CD投影SIP vs 检测星, 距离<3px)
    td=cKDTree(np.column_stack([dx,dy]))
    g_ok=np.isfinite(x_sip)&(x_sip>0)&(x_sip<w)&(y_sip>0)&(y_sip<h)
    gp=np.column_stack([x_sip[g_ok],y_sip[g_ok]]); gi_list=np.where(g_ok)[0]
    ds_all,ids_all=td.query(gp,k=1)
    used=np.zeros(len(dx),dtype=bool)
    pairs=[]
    for kk in np.argsort(ds_all):
        if ds_all[kk]>2:
            break
        ii=ids_all[kk]
        if used[ii]:
            continue
        used[ii]=True
        pairs.append((ii,gi_list[kk]))

    npairs=len(pairs)
    print(f"1对1匹配(<2px): {npairs}对")
    if npairs<10: return

    # 计算每对的残差: src(det) - dst(projected)
    rx_cd=[]; ry_cd=[]; rx_sip=[]; ry_sip=[]
    det_px=[]; det_py=[]
    for di,gi in pairs:
        det_px.append(dx[di]); det_py.append(dy[di])
        rx_cd.append(dx[di]-x_cd[gi]); ry_cd.append(dy[di]-y_cd[gi])
        rx_sip.append(dx[di]-x_sip[gi]); ry_sip.append(dy[di]-y_sip[gi])

    rxc=np.array(rx_cd); ryc=np.array(ry_cd)
    rxs=np.array(rx_sip); rys=np.array(ry_sip)
    dpx=np.array(det_px); dpy=np.array(det_py)

    # 整体统计
    for name,rxi,ryi in [("CD线性",rxc,ryc),("CD+SIP",rxs,rys)]:
        print(f"\n{name}:")
        print(f"  med=[{np.median(rxi):+.3f},{np.median(ryi):+.3f}]px "
              f"mean=[{np.mean(rxi):+.3f},{np.mean(ryi):+.3f}]px "
              f"RMS={np.sqrt(np.mean(rxi**2+ryi**2)):.3f}px")
        # 径向vs切向
        cx,cy=crpix[0],crpix[1]
        radv=np.sqrt((dpx-cx)**2+(dpy-cy)**2)
        rhx=(dpx-cx)/np.maximum(radv,1); rhy=(dpy-cy)/np.maximum(radv,1)
        thx=-rhy; thy=rhx
        rr=rxi*rhx+ryi*rhy; rt=rxi*thx+ryi*thy
        print(f"  径向: med={np.median(rr):+.3f}px mean={np.mean(rr):+.3f}px")
        print(f"  切向: med={np.median(rt):+.3f}px mean={np.mean(rt):+.3f}px")

    # 9区域
    bw,bh=w/3,h/3
    rn=["上左","上中","上右","中左","中中","中右","下左","下中","下右"]
    print("\n--- CD+SIP 9区域 ---")
    sig_patterns=[]
    for ri in range(9):
        rj=ri//3; ci=ri%3
        m=(dpx>=ci*bw)&(dpx<(ci+1)*bw)&(dpy>=rj*bh)&(dpy<(rj+1)*bh)
        if m.sum()<2:
            print(f"  [{rn[ri]}] n<2")
            continue
        mx=np.median(rxs[m]); my=np.median(rys[m])
        ax=np.mean(rxs[m]); ay=np.mean(rys[m])
        ang=math.degrees(math.atan2(my,mx)); mag=math.sqrt(mx*mx+my*my)
        print(f"  [{rn[ri]}] n={int(m.sum()):3d}  med=[{mx:+.3f},{my:+.3f}]px  "
              f"方向={ang:+6.1f}° 幅={mag:.3f}px  mean=[{ax:+.3f},{ay:+.3f}]px")
        sig_patterns.append((ri,rj,ci,ax,ay,mx,my))

    # 检测系统性能量流向
    print("\n--- 模式检测 ---")
    # 9区域 → 9个tuple, 但不满9个时用实际数量
    n_reg = len(sig_patterns)
    if n_reg >= 4:
        all_ax=np.array([p[3] for p in sig_patterns])
        all_ay=np.array([p[4] for p in sig_patterns])
        print(f"{n_reg}区整体均值: [{np.mean(all_ax):+.3f},{np.mean(all_ay):+.3f}]px "
              f"(≈0=无全局平移)")
    else:
        print("区域不足, 跳过模式分析")

    # 渲染
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

    # 残差箭头(下采样)
    step=max(1,npairs//300)
    for k in range(0,npairs,step):
        ax.arrow(dpx[k],dpy[k],-rxs[k],-rys[k],
                 head_width=10,head_length=6,fc='red',ec='red',alpha=0.7,lw=0.2,
                 length_includes_head=True)

    colors=['yellow','cyan','magenta','lime','orange','pink','white','#0ff','#f0f']
    for (ri,rj,ci,axx,ayy,mx,my) in sig_patterns:
        cx_r,cy_r=ci*bw+bw/2,rj*bh+bh/2
        sc=15
        ax.arrow(cx_r,cy_r,-axx*sc,-ayy*sc,head_width=30,head_length=20,
                 fc=colors[ri],ec=colors[ri],alpha=0.9,lw=2,length_includes_head=True)
        ax.text(cx_r,cy_r-35,f'{-axx*sc:+.1f},{-ayy*sc:+.1f}',color=colors[ri],
                fontsize=8,ha='center',bbox=dict(boxstyle='round',facecolor='black',alpha=0.6))

    ax.set_xlim(0,w); ax.set_ylim(0,h); ax.axis("off")
    out=f'{OUT}/_region_{label}.png'
    fig.savefig(out,dpi=DPI,pad_inches=0)
    plt.close(fig)
    print(f"图: {out}")

run("testdata/lights/NGC7293_T2_HO_flying_dutchman-20250607@085204-1200S-H-alpha.fts","NGC7293_Ha")
run("testdata/lights1/panel2/Galaxy_Center_mosaic2_T4_flying_dutchman-20250717@022723-600S-Oiii.fts","GC_P2_Oiii")
run("testdata/lights1/panel1/Galaxy_Center_mosaic1_T4_flying_dutchman-20250813@011752-180S-Red.fts","GC_P1_Red")
