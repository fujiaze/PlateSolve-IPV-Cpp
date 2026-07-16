"""
诊断图: 红圈=检测星, 绿十字=Gaia-WCS投影, 黄线=匹配对连线
每帧抽取200对最亮匹配, 叠加在4张子图上(全图+3个局部放大)
"""
import sys,os,numpy as np,math,json
sys.path.insert(0,'lib/plate_solve/python')
sys.path.insert(0,'lib/astro_image_io/python')
sys.path.insert(0,'lib/star_detector/python')
from astro_image_io import ImageReader
from vector_match_v3_5_cpp import VectorMatchV35Cpp
from vector_match_v2 import GaiaClientPy
from star_detector import StarDetector,SDetParamsPy
from astropy.io import fits as afits
from astropy.coordinates import SkyCoord
import astropy.units as u
from scipy.spatial import cKDTree
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

os.chdir(r"F:\Astro dev\Astro CS Normalization Database")
OUT="overlay_output"

def make_diagnostic(fits_path,label):
    reader=ImageReader();img=reader.read(fits_path)
    w,h=img.width,img.height;fl=img.metadata.observation.focallen
    ps=img.metadata.observation.xpixsz;s0=206.265*ps/fl
    hdul=afits.open(fits_path);hdr=hdul[0].header;hdul.close()
    sc=SkyCoord(hdr.get('RA',''),hdr.get('DEC',''),unit=(u.hourangle,u.deg))
    cra,cdec=sc.ra.deg,sc.dec.deg
    detector=StarDetector(params=SDetParamsPy(fitRadius=0))
    det=detector.detect_ex(img.data)
    dx=np.array(det.x);dy=np.array(det.y)
    df=np.array(det.flux)

    vm=VectorMatchV35Cpp('GaiaDR3')
    js=f'{OUT}/_match_{label}.json'
    result=vm.solve(dx,dy,np.array(det.flux),np.array(det.saturated),
                     cra,cdec,fl,ps,w,h,wcs_out=js)
    vm.close()
    if not result:print(f'{label}:FAIL');return
    with open(js) as f:wc=json.load(f)
    cd=np.array(wc['CD']);crv=np.array(wc['CRVAL']);crp=np.array(wc['CRPIX'])
    sipA=np.array(wc['SIP_A']).reshape(6,6);sipB=np.array(wc['SIP_B']).reshape(6,6)
    so=wc['SIP_ORDER']

    # SIP投影全部Gaia
    fov=math.sqrt(w*w+h*h)*s0/3600.
    gaia=GaiaClientPy('GaiaDR3',1)
    ra_a,dec_a,mag_a=gaia.cone_search(cra,cdec,fov*0.55,22.0);gaia.close()
    ra_a=np.array(ra_a);dec_a=np.array(dec_a);mag_a=np.array(mag_a)
    cdet=cd[0,0]*cd[1,1]-cd[0,1]*cd[1,0]
    cdi=np.array([[cd[1,1],-cd[0,1]],[-cd[1,0],cd[0,0]]])/cdet
    xp=cdi[0,0]*(ra_a-crv[0])+cdi[0,1]*(dec_a-crv[1])
    yp=cdi[1,0]*(ra_a-crv[0])+cdi[1,1]*(dec_a-crv[1])
    xB=xp+crp[0];yB=yp+crp[1]
    mrg=500;ifv=(xB>-mrg)&(xB<w+mrg)&(yB>-mrg)&(yB<h+mrg)
    if ifv.sum()<3:print(f'Gaia不足');return
    xi=xp[ifv].copy();et=yp[ifv].copy();xio=xp[ifv].copy();eto=yp[ifv].copy()
    mo=min(so,6) if so>0 else 0
    sts=[]
    if mo>=2:
        for p in range(mo+1):
            for q in range(mo+1):
                if p+q<2 or p+q>mo:continue
                ac=sipA[p,q];bc=sipB[p,q]
                if abs(ac)>1e-30 or abs(bc)>1e-30:sts.append((p,q,ac,bc))
    for _ in range(20):
        sdx=np.zeros_like(xi);sdy=np.zeros_like(et)
        for p,q,ac,bc in sts:
            xc=np.clip(xi,-5e3,5e3);yc=np.clip(et,-5e3,5e3)
            tm=xc**p*yc**q;tm=np.where(np.isfinite(tm),tm,0)
            sdx+=ac*tm;sdy+=bc*tm
        xn=xio-sdx;yn=eto-sdy
        if np.max(np.abs(xn-xi))<1e-6 and np.max(np.abs(yn-et))<1e-6:break
        xi, et = xn, yn
    xS=np.full(len(ra_a),np.nan);yS=np.full(len(ra_a),np.nan)
    xS[ifv]=xi+crp[0];yS[ifv]=et+crp[1]

    # 1对1互斥匹配 (距离<5px)
    iS=np.isfinite(xS)&(xS>0)&(xS<w)&(yS>0)&(yS<h)
    td=cKDTree(np.column_stack([dx,dy]))
    gp=np.column_stack([xS[iS],yS[iS]]);gil=np.where(iS)[0]
    ds,ids=td.query(gp,k=1)
    used=np.zeros(len(dx),bool);pairs=[]
    for kk in np.argsort(ds):
        if ds[kk]>5:break
        ii=ids[kk]
        if used[ii]:
            continue
        used[ii]=True
        pairs.append((ii,gil[kk]))
    npairs=len(pairs)
    print(f'{label}: {npairs}对 <5px')

    # 按Gaia星等排序, 取最亮200对
    mags=[mag_a[gi] for _,gi in pairs]
    order=np.argsort(mags)
    n_plot=min(200,npairs)
    plot_idx=order[:n_plot]

    # 残差方向统计
    rx_all=[];ry_all=[]
    for kk in plot_idx:
        di,gi=pairs[kk]
        rx_all.append(xS[gi]-dx[di]);ry_all.append(yS[gi]-dy[di])
    rxa=np.array(rx_all);rya=np.array(ry_all)
    print(f'  最亮{n_plot}对 med=[{np.median(rxa):+.2f},{np.median(rya):+.2f}]px mean=[{np.mean(rxa):+.2f},{np.mean(rya):+.2f}]px')

    # 渲染底图
    data=img.data.astype(np.float32)
    dd=data[data>0];lo,hi=np.percentile(dd,(1,99.5)) if len(dd)>1 else (0,1)
    ims=np.clip((data-lo)/max(hi-lo,1),0,1)

    # 4子图: 全图 + 3个局部 (上中, 中中, 下中)
    regions=[(0,0,w,h,"全图"),(w/3,h/3,w/3,h/3,"中中"),
             (w/6,h/6,w/3,h/3,"上中"),(w/6,2*h/3,w/3,h/3,"下中")]
    DPI=100;sc_w=8;sc_h=3
    fig,axes=plt.subplots(1,4,figsize=(sc_w*4,sc_h),dpi=150)
    for ax_idx,(rx,ry,rw,rh,rtitle) in enumerate(regions):
        ax=axes[ax_idx]
        ax.imshow(ims,cmap="gray",origin="lower",interpolation="nearest",
                  extent=[0,w,0,h])
        ax.set_xlim(rx,rx+rw);ax.set_ylim(ry,ry+rh)
        ax.set_title(rtitle,fontsize=9,color='white',y=0.98)
        ax.axis("off")

        # 连线+圆圈+十字(仅帧内点)
        for kk in plot_idx:
            di,gi=pairs[kk]
            pxD,pyD=dx[di],dy[di]
            pxG,pyG=xS[gi],yS[gi]
            if pxD<rx or pxD>rx+rw or pyD<ry or pyD>ry+rh:continue
            # 方向: Gaia→检测 = 残差方向
            res_x=pxD-pxG;res_y=pyD-pyG
            rd=np.sqrt(res_x**2+res_y**2)
            if rd>20:continue
            # 颜色: 红色=残差大, 青=残差小
            t=np.clip(rd/5,0,1)
            color=(t,1-t,0.2) # R→G
            ax.plot([pxG,pxD],[pyG,pyD],'-',color=color,lw=0.5,alpha=0.7)
            # 圆圈(检测星) - 青绿色
            circ=plt.Circle((pxD,pyD),15,fc='none',ec='cyan',lw=0.8,alpha=0.85)
            ax.add_patch(circ)
            # 十字(Gaia投影) - 品红色
            cs=8
            ax.plot([pxG-cs,pxG+cs],[pyG,pyG],'-',color='magenta',lw=0.8,alpha=0.9)
            ax.plot([pxG,pxG],[pyG-cs,pyG+cs],'-',color='magenta',lw=0.8,alpha=0.9)

        # 平均残差大箭头(黄色)
        if len(rxa)>3:
            mdx, mdy = np.mean(rxa), np.mean(rya)
            cx_r, cy_r = rx+rw/2, ry+rh/2
            sc=10
            ax.arrow(cx_r,cy_r,mdx*sc,mdy*sc,head_width=40,head_length=25,
                     fc='yellow',ec='yellow',alpha=0.95,lw=2.5,length_includes_head=True,zorder=99)
            ax.text(cx_r,cy_r-50,f'均值δ=[{mdx:+.2f},{mdy:+.2f}]px',
                    color='yellow',fontsize=10,ha='center',weight='bold',
                    bbox=dict(boxstyle='round',facecolor='black',alpha=0.7))

    # 图例
    fig.text(0.02,0.98,'○=检测星  +=Gaia投影  —=匹配对连线  黄色箭头=均值残差(×10)',
             fontsize=8,color='white',va='top',
             bbox=dict(boxstyle='round',facecolor='black',alpha=0.7))

    out_png=f'{OUT}/_match_{label}.png'
    fig.savefig(out_png,dpi=150,pad_inches=0.1,facecolor='black')
    plt.close(fig)
    print(f'  图: {out_png}')

make_diagnostic("testdata/lights/NGC7293_T2_HO_flying_dutchman-20250607@085204-1200S-H-alpha.fts","NGC7293")
make_diagnostic("testdata/lights1/panel2/Galaxy_Center_mosaic2_T4_flying_dutchman-20250717@022723-600S-Oiii.fts","GC_P2")
make_diagnostic("testdata/lights1/panel1/Galaxy_Center_mosaic1_T4_flying_dutchman-20250813@011752-180S-Red.fts","GC_P1")
