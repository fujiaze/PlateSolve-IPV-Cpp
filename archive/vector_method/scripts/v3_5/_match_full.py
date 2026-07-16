"""全尺寸一张图: 圆圈=检测星, 十字=Gaia投影, 连线=匹配对, 无边框无图例"""
import sys,os
import numpy as np,math,json
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
OUT = "overlay_output"

for fits_path,label in [
    ("testdata/lights/NGC7293_T2_HO_flying_dutchman-20250607@085204-1200S-H-alpha.fts","NGC7293"),
    ("testdata/lights1/panel2/Galaxy_Center_mosaic2_T4_flying_dutchman-20250717@022723-600S-Oiii.fts","GC_P2"),
    ("testdata/lights1/panel1/Galaxy_Center_mosaic1_T4_flying_dutchman-20250813@011752-180S-Red.fts","GC_P1"),
]:
 reader=ImageReader();img=reader.read(fits_path)
 w,h=img.width,img.height;fl=img.metadata.observation.focallen
 ps=img.metadata.observation.xpixsz;s0=206.265*ps/fl
 hdul=afits.open(fits_path);hdr=hdul[0].header;hdul.close()
 sc=SkyCoord(hdr.get('RA',''),hdr.get('DEC',''),unit=(u.hourangle,u.deg))
 cra,cdec=sc.ra.deg,sc.dec.deg
 detector=StarDetector(params=SDetParamsPy(fitRadius=0))
 det=detector.detect_ex(img.data)
 dx=np.array(det.x);dy=np.array(det.y)
 vm=VectorMatchV35Cpp('GaiaDR3')
 js=os.path.join(OUT,f'_match_{label}.json')
 result=vm.solve(dx,dy,np.array(det.flux),np.array(det.saturated),
                  cra,cdec,fl,ps,w,h,wcs_out=js)
 vm.close()
 with open(js) as f:wc=json.load(f)
 cd=np.array(wc['CD']);crv=np.array(wc['CRVAL']);crp=np.array(wc['CRPIX'])
 sipA=np.array(wc['SIP_A']).reshape(6,6);sipB=np.array(wc['SIP_B']).reshape(6,6)
 so=wc['SIP_ORDER']
 fov=math.sqrt(w*w+h*h)*s0/3600.
 gaia=GaiaClientPy('GaiaDR3',1)
 ra_a,dec_a,mag_a=gaia.cone_search(cra,cdec,fov*0.55,22.0);gaia.close()
 ra_a=np.array(ra_a);dec_a=np.array(dec_a);mag_a=np.array(mag_a)
 cdet=cd[0,0]*cd[1,1]-cd[0,1]*cd[1,0]
 cdi=np.array([[cd[1,1],-cd[0,1]],[-cd[1,0],cd[0,0]]])/cdet
 xp=cdi[0,0]*(ra_a-crv[0])+cdi[0,1]*(dec_a-crv[1])
 yp=cdi[1,0]*(ra_a-crv[0])+cdi[1,1]*(dec_a-crv[1])
 xBin=xp+crp[0];yBin=yp+crp[1]
 mrg=500;ifv=(xBin>-mrg)&(xBin<w+mrg)&(yBin>-mrg)&(yBin<h+mrg)
 xi=xp[ifv].copy();et=yp[ifv].copy();xio=xp[ifv].copy();eto=yp[ifv].copy()
 mo=min(so,6) if so>0 else 0
 sts=[]
 if mo>=2:
  for p in range(mo+1):
   for q in range(mo+1):
    if p+q<2 or p+q>mo:continue
    if abs(sipA[p,q])>1e-30 or abs(sipB[p,q])>1e-30:sts.append((p,q,sipA[p,q],sipB[p,q]))
 for _ in range(20):
  sdx=np.zeros_like(xi);sdy=np.zeros_like(et)
  for p,q,ac,bc in sts:
   xc=np.clip(xi,-5e3,5e3);yc=np.clip(et,-5e3,5e3)
   tm=xc**p*yc**q;tm=np.where(np.isfinite(tm),tm,0)
   sdx+=ac*tm;sdy+=bc*tm
  xn=xio-sdx;yn=eto-sdy
  if np.max(np.abs(xn-xi))<1e-6 and np.max(np.abs(yn-et))<1e-6:break
  xi,et=xn,yn
 xS=np.full(len(ra_a),np.nan);yS=np.full(len(ra_a),np.nan)
 xS[ifv]=xi+crp[0];yS[ifv]=et+crp[1]
 iS=np.isfinite(xS)&(xS>0)&(xS<w)&(yS>0)&(yS<h)
 td=cKDTree(np.column_stack([dx,dy]))
 gp=np.column_stack([xS[iS],yS[iS]]);gil=np.where(iS)[0]
 ds,ids=td.query(gp,k=1)
 used=np.zeros(len(dx),bool);pairs=[]
 for kk in np.argsort(ds):
  if ds[kk]>5:break
  ii=ids[kk]
  if used[ii]:continue
  used[ii]=True;pairs.append((ii,gil[kk]))
 npairs=len(pairs)
 print(f"{label}: {npairs}对")
 data=img.data.astype(np.float32)
 dd=data[data>0];lo,hi=np.percentile(dd,(1,99.5)) if len(dd)>1 else (0,1)
 ims=np.clip((data-lo)/max(hi-lo,1),0,1)
 DPI=100
 fig=plt.figure(figsize=(w/DPI,h/DPI),dpi=DPI,frameon=False)
 ax=fig.add_axes([0,0,1,1])
 ax.imshow(ims,cmap="gray",origin="lower",interpolation="nearest")
 for di,gi in pairs:
  pxD,pyD=dx[di],dy[di];pxG,pyG=xS[gi],yS[gi]
  rd=np.sqrt((pxD-pxG)**2+(pyD-pyG)**2)
  t=min(rd/5,1);color=(t,1-t,0.3)
  ax.plot([pxG,pxD],[pyG,pyD],'-',color=color,lw=0.4,alpha=0.6)
  circ=plt.Circle((pxD,pyD),12,fc='none',ec='cyan',lw=0.6,alpha=0.8)
  ax.add_patch(circ)
  cs=6
  ax.plot([pxG-cs,pxG+cs],[pyG,pyG],'-',color='magenta',lw=0.6,alpha=0.85)
  ax.plot([pxG,pxG],[pyG-cs,pyG+cs],'-',color='magenta',lw=0.6,alpha=0.85)
 ax.set_xlim(0,w);ax.set_ylim(0,h);ax.axis("off")
 out_png=os.path.join(OUT,f"_match_full_{label}.png")
 fig.savefig(out_png,dpi=DPI,pad_inches=0)
 plt.close(fig)
 print(f"Saved: {out_png}")
