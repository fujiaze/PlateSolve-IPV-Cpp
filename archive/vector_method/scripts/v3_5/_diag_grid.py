"""
9区域残差方向诊断: 将图像分成3×3网格，计算每个区域匹配对的残差均值向量和方向
检测: 剪切、旋转误差、scale误差、平移
"""
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
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

os.chdir(r"F:\Astro dev\Astro CS Normalization Database")
OUT = "overlay_output"

def solve_and_get_wcs(fits_path):
    reader=ImageReader(); img=reader.read(fits_path)
    w,h=img.width,img.height; fl=img.metadata.observation.focallen
    ps=img.metadata.observation.xpixsz; s0=206.265*ps/fl
    hdul=afits.open(fits_path); hdr=hdul[0].header; hdul.close()
    exptime=float(hdr.get('EXPTIME',1.0))
    sc=SkyCoord(hdr.get('RA',''),hdr.get('DEC',''),unit=(u.hourangle,u.deg))
    cra,cdec=sc.ra.deg,sc.dec.deg
    detector=StarDetector(params=SDetParamsPy(fitRadius=0))
    det=detector.detect_ex(img.data)
    wcs_out = f'{OUT}/_diag_grid.json'
    vm=VectorMatchV35Cpp('GaiaDR3')
    result=vm.solve(np.array(det.x,np.float64),np.array(det.y,np.float64),
                     np.array(det.flux,np.float64),np.array(det.saturated,np.int32),
                     cra,cdec,fl,ps,w,h,wcs_out=wcs_out,skip_sip=False,exptime=exptime)
    vm.close()
    if not result: return None
    with open(wcs_out) as f: wcs=json.load(f)
    return img,det,result,wcs,s0,w,h,cra,cdec

def wcs_sip_project(ra_src,dec_src,cd,crval,crpix,sip_A,sip_B,sip_order,w,h):
    cdet=cd[0,0]*cd[1,1]-cd[0,1]*cd[1,0]
    if abs(cdet)<1e-30: return np.full(len(ra_src),np.nan),np.full(len(ra_src),np.nan)
    cdi=np.array([[cd[1,1],-cd[0,1]],[-cd[1,0],cd[0,0]]])/cdet
    xi_p=cdi[0,0]*(ra_src-crval[0])+cdi[0,1]*(dec_src-crval[1])
    et_p=cdi[1,0]*(ra_src-crval[0])+cdi[1,1]*(dec_src-crval[1])
    x_lin=xi_p+crpix[0]; y_lin=et_p+crpix[1]
    mrg=500; ifv=(x_lin>-mrg)&(x_lin<w+mrg)&(y_lin>-mrg)&(y_lin<h+mrg)
    xi=xi_p[ifv].copy(); et=et_p[ifv].copy()
    xio=xi_p[ifv].copy(); eto=et_p[ifv].copy()
    mo=min(sip_order,6) if sip_order>0 else 0
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
    xp=np.full(len(ra_src),np.nan); yp=np.full(len(ra_src),np.nan)
    xp[ifv]=xi+crpix[0]; yp[ifv]=et+crpix[1]
    return xp,yp

def region_analysis(det_x,det_y,x_gaia,y_gaia,w,h,label):
    """将图像分为3×3区域，分析每个区域Gaia→检测星的残差向量"""
    bw, bh = w/3, h/3

    td = cKDTree(np.column_stack([det_x,det_y]))
    gaia_ok = np.isfinite(x_gaia)&(x_gaia>0)&(x_gaia<w)&(y_gaia>0)&(y_gaia<h)
    gp = np.column_stack([x_gaia[gaia_ok],y_gaia[gaia_ok]])
    ds, ids = td.query(gp,k=1)
    match_ok = ds < max(2,2*206.265*6/200)  # 2px or equivalent

    rx = gp[match_ok,0]-det_x[ids[match_ok]]
    ry = gp[match_ok,1]-det_y[ids[match_ok]]
    gx = gp[match_ok,0]; gy = gp[match_ok,1]

    reg = np.zeros(len(rx),dtype=int)
    for k,(gxx,gyy) in enumerate(zip(gx,gy)):
        ri = int(gyy//bh); ci = int(gxx//bw)
        ri = max(0,min(2,ri)); ci = max(0,min(2,ci))
        reg[k] = ri*3+ci

    print(f"\n{'='*70}")
    print(f"{label}: {w}x{h} (region={bw:.0f}x{bh:.0f}px)")
    print(f"匹配对: {match_ok.sum()} (距离<2px)")

    # Row labels: top/bottom in standard CRVAL view, top in image=row2
    row_names = ["上左","上中","上右","中左","中中","中右","下左","下中","下右"]

    results = {}
    for ri in range(9):
        m = reg==ri
        if m.sum()<2:
            print(f"  [{row_names[ri]}] 星点不足")
            continue
        mx, my = np.median(rx[m]), np.median(ry[m])
        ax, ay = np.mean(rx[m]), np.mean(ry[m])
        sdx, sdy = np.std(rx[m]), np.std(ry[m])
        ang = math.degrees(math.atan2(my,mx))
        mag = math.sqrt(mx*mx+my*my)
        print(f"  [{row_names[ri]}] n={m.sum():4d}  med=[{mx:+.3f},{my:+.3f}]px mean=[{ax:+.3f},{ay:+.3f}]px  "
              f"方向={ang:+.1f}° 幅={mag:.3f}px  std=[{sdx:.3f},{sdy:.3f}]")
        results[ri] = {"n":int(m.sum()),"med_x":mx,"med_y":my,"mean_x":ax,"mean_y":ay,
                       "std_x":sdx,"std_y":sdy,"angle":ang,"mag":mag}

    # 整体统计
    print(f"\n整体: med=[{np.median(rx):+.3f},{np.median(ry):+.3f}]px  "
          f"mean=[{np.mean(rx):+.3f},{np.mean(ry):+.3f}]px  "
          f"std=[{np.std(rx):.3f},{np.std(ry):.3f}]px  RMS={np.sqrt(np.mean(rx**2+ry**2)):.3f}px")

    # 计算径向分量和切向分量（相对图像中心）
    cx, cy = w/2, h/2
    rad = np.sqrt((gx-cx)**2+(gy-cy)**2)
    r_hat_x = (gx-cx)/np.maximum(rad,1)
    r_hat_y = (gy-cy)/np.maximum(rad,1)
    t_hat_x = -r_hat_y
    t_hat_y = r_hat_x
    rad_res = rx*r_hat_x + ry*r_hat_y
    tan_res = rx*t_hat_x + ry*t_hat_y
    print(f"径向残差: med={np.median(rad_res):+.3f}px mean={np.mean(rad_res):+.3f}px "
          f"(+向外=scale偏小, -向内=scale偏大)")
    print(f"切向残差: med={np.median(tan_res):+.3f}px mean={np.mean(tan_res):+.3f}px "
          f"(±=旋转误差)")

    # 检测shear: 对角区域残差方向是否相反
    if 0 in results and 8 in results:
        m0,m8=results[0]["med_x"]+results[8]["med_x"],results[0]["med_y"]+results[8]["med_y"]
        print(f"对角线(上左+下右)残差和: [{m0:+.3f},{m8:+.3f}]px (≈0=无shear)")
    if 2 in results and 6 in results:
        m2,m6=results[2]["med_x"]+results[6]["med_x"],results[2]["med_y"]+results[6]["med_y"]
        print(f"对角线(上右+下左)残差和: [{m2:+.3f},{m6:+.3f}]px (≈0=无shear)")

    return gx,gy,rx,ry,reg,results

def plot_residuals(img_data, w, h, label, gx, gy, rx, ry, reg):
    """绘制残差向量覆盖图：箭头从检测星指向Gaia投影点"""
    data = img_data.astype(np.float32)
    dd = data[data>0]
    lo,hi = np.percentile(dd,(1,99.5)) if len(dd)>1 else (0,1)
    ims = np.clip((data-lo)/max(hi-lo,1),0,1)
    DPI=100
    fig=plt.figure(figsize=(w/DPI,h/DPI),dpi=DPI,frameon=False)
    ax=fig.add_axes([0,0,1,1])
    ax.imshow(ims,cmap="gray",origin="lower",interpolation="nearest")

    # 画3x3格线
    bw,bh=w/3,h/3
    for i in range(1,3):
        ax.axvline(i*bw,color='cyan',lw=0.5,alpha=0.4)
        ax.axhline(i*bh,color='cyan',lw=0.5,alpha=0.4)

    # 下采样残差箭头（太多会看不清）
    step = max(1,len(rx)//300)
    for k in range(0,len(rx),step):
        ax.arrow(gx[k]-rx[k],gy[k]-ry[k],rx[k],ry[k],
                 head_width=12,head_length=8,fc='red',ec='red',alpha=0.6,lw=0.3,
                 length_includes_head=True)

    # 各区域平均残差
    bw,bh=w/3,h/3
    colors=['yellow','cyan','magenta','lime','orange','pink','white','#0ff','#f0f']
    for ri in range(9):
        m=reg==ri
        if m.sum()<2: continue
        mx,my=np.mean(rx[m]),np.mean(ry[m])
        cx_r=(ri%3)*bw+bw/2; cy_r=(ri//3)*bh+bh/2
        scale=10
        ax.arrow(cx_r,cy_r,mx*scale,my*scale,
                 head_width=30,head_length=20,fc=colors[ri],ec=colors[ri],
                 alpha=0.95,lw=2,length_includes_head=True)
        ax.text(cx_r,cy_r-40,f'{mx*scale:+.1f},{my*scale:+.1f}',
                color=colors[ri],fontsize=8,ha='center',va='top',
                bbox=dict(boxstyle='round',facecolor='black',alpha=0.6))

    ax.set_xlim(0,w); ax.set_ylim(0,h); ax.axis("off")
    out=f'{OUT}/_residual_{label}.png'
    fig.savefig(out,dpi=DPI,pad_inches=0)
    plt.close(fig)
    print(f"残差图: {out}")

# ====== 运行诊断 ======
frames = [
    ("testdata/lights/NGC7293_T2_HO_flying_dutchman-20250607@085204-1200S-H-alpha.fts","NGC7293_Ha"),
    ("testdata/lights1/panel2/Galaxy_Center_mosaic2_T4_flying_dutchman-20250717@022723-600S-Oiii.fts","GC_P2_Oiii"),
    ("testdata/lights1/panel1/Galaxy_Center_mosaic1_T4_flying_dutchman-20250813@011752-180S-Red.fts","GC_P1_Red"),
]

for fits_path, label in frames:
    r = solve_and_get_wcs(fits_path)
    if r is None: print(f"{label}: FAIL"); continue
    img,det,result,wcs,s0,w,h,cra,cdec = r
    cd=np.array(wcs['CD'],np.float64); crval=np.array(wcs['CRVAL'],np.float64)
    crpix=np.array(wcs['CRPIX'],np.float64)
    sip_A=np.array(wcs.get('SIP_A',[0]*36),np.float64).reshape(6,6)
    sip_B=np.array(wcs.get('SIP_B',[0]*36),np.float64).reshape(6,6)
    sip_o=wcs.get('SIP_ORDER',0)

    # 查询Gaia星
    fov_d=math.sqrt(w*w+h*h)*s0/3600.0
    gaia=GaiaClientPy('GaiaDR3',1)
    ra_a,dec_a,_=gaia.cone_search(cra,cdec,fov_d*0.55,22.0)
    gaia.close()
    ra_a=np.array(ra_a,np.float64); dec_a=np.array(dec_a,np.float64)

    x_g,y_g=wcs_sip_project(ra_a,dec_a,cd,crval,crpix,sip_A,sip_B,sip_o,w,h)
    gx,gy,rx,ry,reg,results = region_analysis(
        np.array(det.x,np.float64),np.array(det.y,np.float64),
        x_g,y_g,w,h,label)
    plot_residuals(img.data,w,h,label,gx,gy,rx,ry,reg)
