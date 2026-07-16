#include "psolve_coarse.h"
#include "psolve_fov.h"
#include "psolve_projection.h"
#include "psolve_triangle.h"
#include "psolve_log.h"
#include "gaia_client.h"
#include <stdlib.h>
#include <string.h>
#include <math.h>

struct PSolveHandle_s {
    void *gaia_client;
    double last_scale;
    PSolveAffine last_affine;
    PSolveCoarseResult last_coarse;
    int has_coarse;
    PSolveWCS last_wcs;
    int has_wcs;
    PSolveResult last_result;
    int has_result;
};

typedef struct PSolveHandle_s PSolveHandleInternal;

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif
#define DEG2RAD (M_PI/180.0)

#define NB         60
#define RADIUS     0.002
#define SMIN       0.3
#define SMAX       3.0
#define MATCH_R    50.0
#define RMS_R      10.0
#define STARTN     6
#define CMAX       5
#define CONV_PX    0.02
#define PCT_SCALE  30.0
#define GCS         50.0
#define GMAX        2000

typedef struct {int*h;int*n;int gw,gh;double x0,y0;}Grid;
typedef struct {double ra,dec;float mag;}Gs;

static int cg(const void*a,const void*b){float ma=((const Gs*)a)->mag,mb=((const Gs*)b)->mag;return(ma>mb)-(ma<mb);}
static void sg(double*ra,double*dec,float*mag,int n){if(n<=1)return;Gs*s=(Gs*)malloc(n*sizeof(Gs));for(int i=0;i<n;i++){s[i].ra=ra[i];s[i].dec=dec[i];s[i].mag=mag[i];}qsort(s,n,sizeof(Gs),cg);for(int i=0;i<n;i++){ra[i]=s[i].ra;dec[i]=s[i].dec;mag[i]=s[i].mag;}free(s);}
static void ap(const double*x,const double*y,int n,const double a[6],double*u,double*v){for(int i=0;i<n;i++){u[i]=a[0]+a[1]*x[i]+a[2]*y[i];v[i]=a[3]+a[4]*x[i]+a[5]*y[i];}}
static void rt(double x,double y,int m,double*ox,double*oy){switch(m){case 0:*ox=x;*oy=y;break;case 1:*ox=-x;*oy=-y;break;case 2:*ox=-x;*oy=y;break;case 3:*ox=x;*oy=-y;break;}}

static int fls(const double*x,const double*y,const double*u,const double*v,int n,double a[6]){
    if(n<3)return 1;
    double sx=0,sy=0,su=0,sv=0,sx2=0,sy2=0,sxy=0,sxu=0,syu=0,sxv=0,syv=0;
    for(int i=0;i<n;i++){double xi=x[i],yi=y[i],ui=u[i],vi=v[i];sx+=xi;sy+=yi;su+=ui;sv+=vi;sx2+=xi*xi;sy2+=yi*yi;sxy+=xi*yi;sxu+=xi*ui;syu+=yi*ui;sxv+=xi*vi;syv+=yi*vi;}
    double d=(double)n*(sx2*sy2-sxy*sxy)-sx*(sx*sy2-sy*sxy)+sy*(sx*sxy-sy*sx2);
    if(fabs(d)<1e-20)return 1;double inv=1.0/d;
    a[0]=inv*((sx2*sy2-sxy*sxy)*su-(sx*sy2-sy*sxy)*sxu+(sx*sxy-sy*sx2)*syu);
    a[1]=inv*(-(sx*sy2-sy*sxy)*su+((double)n*sy2-sy*sy)*sxu-((double)n*sxy-sx*sy)*syu);
    a[2]=inv*((sx*sxy-sy*sx2)*su-((double)n*sxy-sx*sy)*sxu+((double)n*sx2-sx*sx)*syu);
    a[3]=inv*((sx2*sy2-sxy*sxy)*sv-(sx*sy2-sy*sxy)*sxv+(sx*sxy-sy*sx2)*syv);
    a[4]=inv*(-(sx*sy2-sy*sxy)*sv+((double)n*sy2-sy*sy)*sxv-((double)n*sxy-sx*sy)*syv);
    a[5]=inv*((sx*sxy-sy*sx2)*sv-((double)n*sxy-sx*sy)*sxv+((double)n*sx2-sx*sx)*syv);
    return 0;
}

static void iva(const double a[6],double inv[6]){
    double dt=a[1]*a[5]-a[2]*a[4];if(fabs(dt)<1e-20){memset(inv,0,6*sizeof(double));return;}
    inv[0]=(a[2]*a[3]-a[0]*a[5])/dt;inv[1]=a[5]/dt;inv[2]=-a[2]/dt;
    inv[3]=(a[0]*a[4]-a[1]*a[3])/dt;inv[4]=-a[4]/dt;inv[5]=a[1]/dt;
}

static void bg(const double*x,const double*y,int n,double x0,double x1,double y0,double y1,Grid*g){
    g->gw=(int)((x1-x0)/GCS)+1;g->gh=(int)((y1-y0)/GCS)+1;int tc=g->gw*g->gh;g->x0=x0;g->y0=y0;
    g->h=(int*)malloc(tc*sizeof(int));g->n=(int*)malloc(n*sizeof(int));
    for(int i=0;i<tc;i++)g->h[i]=-1;
    for(int i=0;i<n;i++){int cx=(int)((x[i]-x0)/GCS),cy=(int)((y[i]-y0)/GCS);if(cx<0)cx=0;if(cx>=g->gw)cx=g->gw-1;if(cy<0)cy=0;if(cy>=g->gh)cy=g->gh-1;g->n[i]=g->h[cy*g->gw+cx];g->h[cy*g->gw+cx]=i;}
}
static void fg(Grid*g){free(g->h);free(g->n);}
static int gn(const Grid*g,const double*x,const double*y,double qx,double qy,double md){
    int cx=(int)((qx-g->x0)/GCS),cy=(int)((qy-g->y0)/GCS);double b=md;int bi=-1;
    for(int dy=-1;dy<=1;dy++)for(int dx=-1;dx<=1;dx++){int nx=cx+dx,ny=cy+dy;if(nx<0||nx>=g->gw||ny<0||ny>=g->gh)continue;
        for(int k=g->h[ny*g->gw+nx];k!=-1;k=g->n[k]){double d2=(x[k]-qx)*(x[k]-qx)+(y[k]-qy)*(y[k]-qy);if(d2<b){b=d2;bi=k;}}}
    return bi;
}

static int af(const double*dx,const double*dy,int nd,const double*cx,const double*cy,int nc,double a[6],int**o_ma,int**o_mb,int*o_nm){
    PSolveTriangle*dt=NULL;int dtc=0;psolve_build_triangles(dx,dy,nd,NB,&dt,&dtc);
    PSolveTriangle*ct=NULL;int ctc=0;psolve_build_triangles(cx,cy,nc,NB,&ct,&ctc);
    PSolveStarPair*pr=NULL;int np=0;
    int r=psolve_match_triangles(dt,dtc,ct,ctc,RADIUS,SMIN,SMAX,&pr,&np);
    psolve_free_triangles(dt);psolve_free_triangles(ct);
    if(r||np<STARTN){if(pr)psolve_free_pairs(pr);*o_ma=NULL;*o_mb=NULL;return-1;}
    double*px=(double*)malloc(np*sizeof(double)),*py=(double*)malloc(np*sizeof(double));
    double*ix=(double*)malloc(np*sizeof(double)),*iy=(double*)malloc(np*sizeof(double));
    for(int i=0;i<np;i++){ix[i]=dx[pr[i].img_idx];iy[i]=dy[pr[i].img_idx];px[i]=cx[pr[i].cat_idx];py[i]=cy[pr[i].cat_idx];}
    r=fls(ix,iy,px,py,np,a);
    free(px);free(py);free(ix);free(iy);psolve_free_pairs(pr);
    if(r){*o_ma=NULL;*o_mb=NULL;return-1;}
    int*ma=(int*)malloc(nd*sizeof(int)),*mb=(int*)malloc(nd*sizeof(int));int nm=0;
    double*tx=(double*)malloc(nd*sizeof(double)),*ty=(double*)malloc(nd*sizeof(double));
    ap(dx,dy,nd,a,tx,ty);
    double mr2=MATCH_R*MATCH_R;
    for(int j=0;j<nd;j++){double b2=1e30;int bj=-1;
        for(int i=0;i<nc;i++){double d2=(cx[i]-tx[j])*(cx[i]-tx[j])+(cy[i]-ty[j])*(cy[i]-ty[j]);if(d2<b2){b2=d2;bj=i;}}
        if(b2<mr2){ma[nm]=j;mb[nm]=bj;nm++;}}
    free(tx);free(ty);
    if(nm<STARTN){free(ma);free(mb);*o_ma=NULL;*o_mb=NULL;return-1;}
    double*ltx=(double*)malloc(nm*sizeof(double)),*lty=(double*)malloc(nm*sizeof(double));
    double*lix=(double*)malloc(nm*sizeof(double)),*liy=(double*)malloc(nm*sizeof(double));
    for(int i=0;i<nm;i++){lix[i]=dx[ma[i]];liy[i]=dy[ma[i]];ltx[i]=cx[mb[i]];lty[i]=cy[mb[i]];}
    if(!fls(lix,liy,ltx,lty,nm,a)){*o_ma=ma;*o_mb=mb;*o_nm=nm;free(ltx);free(lty);free(lix);free(liy);return nm;}
    free(ma);free(mb);free(ltx);free(lty);free(lix);free(liy);*o_ma=NULL;*o_mb=NULL;return-1;
}

int psolve_coarse_solve(void*handle,const uint16_t*im,int w,int ht,const PSolveImageData*id,const double*dx,const double*dy,int nd,PSolveCoarseResult*rs){
    PSolveHandleInternal*h=(PSolveHandleInternal*)handle;
    (void)im;
    PSLOG_I("===Coarse:%dx%d %dd===",w,ht,nd);
    if(nd<3){PSLOG_E("fewstars");return PSOLVE_ERR_NOT_ENOUGH;}
    double sca=psolve_compute_scale(id->focal_length_mm,id->pixel_size_um);
    double fw=psolve_compute_fov_w(sca,w),fh=psolve_compute_fov_h(sca,ht),fr=psolve_compute_fov_radius(sca,w,ht);h->last_scale=sca;
    PSLOG_I("sc=%.3f\"/px FOV=%.0f'x%.0f' r=%.4f",sca,fw,fh,fr);
    double hw=w/2.0,hh=ht/2.0,dp=3600.0/sca;
    double ra=id->center_ra,dec=id->center_dec;
    double*dc=(double*)malloc(nd*sizeof(double)),*dc2=(double*)malloc(nd*sizeof(double));
    for(int i=0;i<nd;i++){dc[i]=dx[i]-hw;dc2[i]=hh-dy[i];}
    
    double initial_mag=psolve_estimate_mag_limit(id->focal_length_mm,id->exposure_time_s);
    PSLOG_I("Initial estimated mag=%.2f",initial_mag);
    
    int target_stars=nd*1.5;
    double mm=initial_mag;
    int ac=0;
    
    if(target_stars>0){
        mm=psolve_bisection_mag_limit(h->gaia_client,ra,dec,fr+5.0,target_stars,&mm,&ac);
    }
    
    double*ar=NULL,*ad=NULL;float*am=NULL;
    int rc=gaia_client_cone_search_for_solver((GaiaClient*)h->gaia_client,ra,dec,fr+5.0,mm,&ar,&ad,&am,&ac);
    if(rc||ac==0){free(dc);free(dc2);PSLOG_E("Gaiaempty");return PSOLVE_ERR_NO_MATCH;}
    PSLOG_I("Gaia:%d stars (mag_limit=%.2f)",ac,mm);sg(ar,ad,am,ac);
    int tc=ac<GMAX?ac:GMAX;

    int bm=-1,bnm=0,bnp=0;double ba[6];int*bma=NULL,*bmb=NULL;
    for(int mode=0;mode<4;mode++){
        double*cx2=NULL,*cy2=NULL;
        psolve_project_stars(ar,ad,tc,ra,dec,&cx2,&cy2);
        for(int i=0;i<tc;i++){cx2[i]*=dp;cy2[i]*=dp;}
        double*rx=(double*)malloc(tc*sizeof(double)),*ry=(double*)malloc(tc*sizeof(double));
        for(int i=0;i<tc;i++)rt(cx2[i],cy2[i],mode,&rx[i],&ry[i]);
        double a[6];int*ma=NULL,*mb=NULL;int nm;
        int np=af(dc,dc2,nd,rx,ry,tc,a,&ma,&mb,&nm);
        free(rx);free(ry);free(cx2);free(cy2);
        if(np>=STARTN&&nm>bnm){bnm=nm;bm=mode;memcpy(ba,a,sizeof(a));free(bma);free(bmb);bma=ma;bmb=mb;}
        else{free(ma);free(mb);}}
    if(bm<0){free(dc);free(dc2);free(ar);free(ad);free(am);PSLOG_E("nohypo");return PSOLVE_ERR_NO_MATCH;}
    PSLOG_I("best:mode=%d nm=%d sc=%.3f",bm,bnm,sqrt(ba[1]*ba[1]+ba[4]*ba[4]));

    double tr[6];memcpy(tr,ba,sizeof(tr));
    double off=sqrt(tr[0]*tr[0]+tr[3]*tr[3]);
    double nra,nde;psolve_plane_to_sky(tr[0]/dp,tr[3]/dp,ra,dec,&nra,&nde);
    ra=nra;dec=nde;
    int trial=0;
    PSLOG_I("it0:off=%.1f\" c=(%.6f,%.6f)",off,ra,dec);

    double*ctx=(double*)malloc(tc*sizeof(double)),*cty=(double*)malloc(tc*sizeof(double));
    while(off*sca>0.01&&trial<CMAX){trial++;
        psolve_project_stars(ar,ad,tc,ra,dec,&ctx,&cty);
        for(int i=0;i<tc;i++){ctx[i]*=dp;cty[i]*=dp;}
        double*rxa=(double*)malloc(tc*sizeof(double)),*rya=(double*)malloc(tc*sizeof(double));
        for(int i=0;i<tc;i++)rt(ctx[i],cty[i],bm,&rxa[i],&rya[i]);
        double*lcx=(double*)malloc(bnm*sizeof(double)),*lcy=(double*)malloc(bnm*sizeof(double));
        for(int i=0;i<bnm;i++){lcx[i]=rxa[bmb[i]];lcy[i]=rya[bmb[i]];}
        free(rxa);free(rya);
        double*lix=(double*)malloc(bnm*sizeof(double)),*liy=(double*)malloc(bnm*sizeof(double));
        for(int i=0;i<bnm;i++){lix[i]=dc[bma[i]];liy[i]=dc2[bma[i]];}
        fls(lix,liy,lcx,lcy,bnm,tr);
        free(lcx);free(lcy);free(lix);free(liy);
        off=sqrt(tr[0]*tr[0]+tr[3]*tr[3]);
        psolve_plane_to_sky(tr[0]/dp,tr[3]/dp,ra,dec,&nra,&nde);
        ra=nra;dec=nde;
        PSLOG_I("it%d:off=%.1f\" c=(%.6f,%.6f)",trial,off*sca,ra,dec);
        if(off*sca<0.01){PSLOG_I("cvrg");break;}}
    free(ctx);free(cty);

    double scl=sqrt(tr[1]*tr[1]+tr[4]*tr[4]),sv=fabs(fabs(tr[1])-fabs(tr[5]))+fabs(fabs(tr[2])-fabs(tr[4]));
    PSLOG_I("fin:c=(%.6f,%.6f) sc=%.4f san=%.4f nr=%d",ra,dec,scl,sv,bnm);

    double inv[6];iva(tr,inv);
    int rct=nd<ac?nd:ac;float rmm=am[rct-1];
    int cnt=0;for(int i=0;i<ac;i++)if(am[i]<=rmm)cnt++;
    double*rra=(double*)malloc(cnt*sizeof(double)),*rde=(double*)malloc(cnt*sizeof(double));
    float*rm=(float*)malloc(cnt*sizeof(float));
    double*px=(double*)malloc(cnt*sizeof(double)),*py=(double*)malloc(cnt*sizeof(double));
    int ki=0;for(int i=0;i<ac;i++)if(am[i]<=rmm){rra[ki]=ar[i];rde[ki]=ad[i];rm[ki]=am[i];ki++;}
    for(int i=0;i<cnt;i++){psolve_sky_to_plane(rra[i],rde[i],ra,dec,&px[i],&py[i]);px[i]*=dp;py[i]*=dp;}
    double*rxx=(double*)malloc(cnt*sizeof(double)),*ryy=(double*)malloc(cnt*sizeof(double));
    for(int i=0;i<cnt;i++)rt(px[i],py[i],bm,&rxx[i],&ryy[i]);
    double*aix=(double*)malloc(cnt*sizeof(double)),*aiy=(double*)malloc(cnt*sizeof(double));
    ap(rxx,ryy,cnt,inv,aix,aiy);

    double mnx=dc[0],mxx=dc[0],mny=dc2[0],mxy=dc2[0];
    for(int i=1;i<nd;i++){if(dc[i]<mnx)mnx=dc[i];if(dc[i]>mxx)mxx=dc[i];if(dc2[i]<mny)mny=dc2[i];if(dc2[i]>mxy)mxy=dc2[i];}
    Grid g;bg(dc,dc2,nd,mnx,mxx,mny,mxy,&g);
    PSolveMatchedStar*ms=(PSolveMatchedStar*)malloc(cnt*sizeof(PSolveMatchedStar));
    int mp=0;double sx=0,sy=0,lm2=RMS_R*RMS_R;
    for(int i=0;i<cnt;i++){int bi=gn(&g,dc,dc2,aix[i],aiy[i],lm2);
        if(bi>=0){double rxxx=dc[bi]-aix[i],ryyy=dc2[bi]-aiy[i];
        ms[mp].img_x=dc[bi];ms[mp].img_y=dc2[bi];ms[mp].cat_ra=rra[i];ms[mp].cat_dec=rde[i];
        ms[mp].cat_mag=rm[i];ms[mp].residual_x=rxxx;ms[mp].residual_y=ryyy;sx+=rxxx*rxxx;sy+=ryyy*ryyy;mp++;}}
    fg(&g);free(rxx);free(ryy);free(aix);free(aiy);free(rra);free(rde);free(rm);free(px);free(py);free(bma);free(bmb);

    double rst=mp>0?sqrt((sx+sy)/mp):0;
    PSLOG_I("RMS:%.4fpx(%d)",rst,mp);
    rs->scale_arcsec_px=sca;rs->fov_w_arcmin=fw;rs->fov_h_arcmin=fh;rs->fov_radius_deg=fr;
    rs->limit_mag=mm;rs->gaia_star_count=cnt;rs->detected_star_count=nd;
    rs->affine.a0=inv[0];rs->affine.a1=inv[1];rs->affine.a2=inv[2];
    rs->affine.b0=inv[3];rs->affine.b1=inv[4];rs->affine.b2=inv[5];
    rs->rms_x=mp>0?sqrt(sx/mp):0;rs->rms_y=mp>0?sqrt(sy/mp):0;rs->rms_total=rst;
    rs->iteration_count=trial;rs->matched_count=mp;rs->matched_stars=ms;
    PSLOG_I("===done===");
    free(dc);free(dc2);free(ar);free(ad);free(am);
    return(mp>=STARTN)?PSOLVE_OK:PSOLVE_ERR_NO_MATCH;
}
