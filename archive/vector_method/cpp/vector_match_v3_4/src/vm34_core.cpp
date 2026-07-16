/**
 * vm34_core.cpp - V3.4 Record-and-Filter + Expand + Clean + SIP
 *
 * Phase A: 无放回1点抽样, s-in-range, θ加权直方图, θ_SNR停止
 * Phase B: θ峰值+n_in_range双过滤 → 1对1 → Umeyama SVD → 初始变换
 * Phase C: 复用Phase A records → 精确变换验证 → 扩充对应关系
 * Phase D: 迭代中位数+3σ剔除 → 收敛到刚体变换 + 干净对应
 * Phase E: 一次性最小二乘6阶多项式 → CD矩阵 + SIP系数
 *
 * Eigen3, nanoflann, C++17, OpenMP
 */

#include <cstdio>
#include <cmath>
#include <vector>
#include <array>
#include <algorithm>
#include <numeric>
#include <random>
#include <utility>
#include <cstring>
#include <atomic>
#include <unordered_set>
#include <omp.h>

#include "../include/vm34_api.h"
#include "Eigen/Dense"
#include "nanoflann.hpp"

namespace vm34 {

static constexpr double PI = 3.14159265358979323846;
static constexpr double DEGTORAD = PI / 180.0;
static constexpr double RADTODEG = 180.0 / PI;

struct PointCloud2D {
    std::vector<std::array<double, 2>> pts;
    inline size_t kdtree_get_point_count() const { return pts.size(); }
    inline double kdtree_get_pt(size_t idx, size_t dim) const { return pts[idx][dim]; }
    template <class BBOX> bool kdtree_get_bbox(BBOX&) const { return false; }
};

using KDTree = nanoflann::KDTreeSingleIndexAdaptor<
    nanoflann::L2_Simple_Adaptor<double, PointCloud2D>, PointCloud2D, 2>;
using KDTreeIndexType = KDTree::IndexType;

// --- basic geometry ---
void apply_similarity(const double* W, int M, double s, double theta,
                      double tx, double ty, double* Wt) {
    double ct = std::cos(theta), st = std::sin(theta);
    for (int i = 0; i < M; ++i) {
        double wx = W[i*2], wy = W[i*2+1];
        Wt[i*2]     = s*(ct*wx - st*wy) + tx;
        Wt[i*2+1]   = s*(st*wx + ct*wy) + ty;
    }
}

void apply_flip(const double* W, int M, int mode, double* Wf) {
    bool fx = (mode==1 || mode==3), fy = (mode==2 || mode==3);
    for (int i=0; i<M; ++i) { Wf[i*2]=fx?-W[i*2]:W[i*2]; Wf[i*2+1]=fy?-W[i*2+1]:W[i*2+1]; }
}

static inline double angle_diff_deg(double a, double b) {
    double d=std::fmod(std::fmod(a-b+180.0,360.0)+360.0,360.0)-180.0;
    return std::abs(d);
}
static inline double wrap180(double d) { return std::fmod(std::fmod(d+180.0,360.0)+360.0,360.0)-180.0; }

// --- median ---
double vec_median(std::vector<double>& v) {
    size_t n=v.size(); if(n==0) return 0;
    std::nth_element(v.begin(), v.begin()+n/2, v.end());
    if(n%2==0){ std::nth_element(v.begin(), v.begin()+n/2-1, v.end()); return (v[n/2]+v[n/2-1])*0.5; }
    return v[n/2];
}

// --- s-in-range ---
int count_s_in_range(const double* U, int N, const double* Wt, int M,
                      const double* norm_U, const double* norm_Wf,
                      double s_min, double s_max, double max_dist) {
    if(N==0||M==0) return 0;
    PointCloud2D cloud; cloud.pts.resize(M);
    for(int i=0;i<M;++i) cloud.pts[i]={Wt[i*2],Wt[i*2+1]};
    KDTree tree(2, cloud, nanoflann::KDTreeSingleIndexAdaptorParams(10));
    double d2max=max_dist*max_dist; int c=0;
    for(int k=0;k<N;++k){
        double q[2]={U[k*2],U[k*2+1]}; KDTreeIndexType idx; double ds;
        nanoflann::KNNResultSet<double,KDTreeIndexType> rs(1); rs.init(&idx,&ds);
        tree.findNeighbors(rs,q);
        if(ds>d2max) continue;
        if(norm_U[k]/norm_Wf[idx]>=s_min && norm_U[k]/norm_Wf[idx]<=s_max) c++;
    }
    return c;
}

// --- count_inliers_1to1 ---
struct InlierResult { int n_inliers; double rms; std::vector<int> inlier_mask; };
InlierResult count_inliers_1to1(const double* U, int N, const double* Wt, int M, double tau) {
    InlierResult r; r.n_inliers=0; r.rms=0; r.inlier_mask.assign(N,0);
    if(N==0||M==0) return r;
    PointCloud2D cloud; cloud.pts.resize(M);
    for(int i=0;i<M;++i) cloud.pts[i]={Wt[i*2],Wt[i*2+1]};
    KDTree tree(2,cloud,nanoflann::KDTreeSingleIndexAdaptorParams(10));
    struct Match { int u,w; double d; };
    std::vector<Match> cand; cand.reserve(N); double t2=tau*tau;
    for(int i=0;i<N;++i){
        double q[2]={U[i*2],U[i*2+1]}; KDTreeIndexType idx; double ds;
        nanoflann::KNNResultSet<double,KDTreeIndexType> rs(1); rs.init(&idx,&ds);
        tree.findNeighbors(rs,q);
        if(ds<=t2) cand.push_back({i,(int)idx,std::sqrt(ds)});
    }
    std::sort(cand.begin(),cand.end(),[](const Match&a,const Match&b){return a.d<b.d;});
    std::vector<int> wu(M,0); double ss=0;
    for(auto&c:cand){if(wu[c.w])continue;wu[c.w]=1;r.inlier_mask[c.u]=1;ss+=c.d*c.d;r.n_inliers++;}
    if(r.n_inliers>0) r.rms=std::sqrt(ss/r.n_inliers);
    return r;
}

double compute_normalized_score(int n, double rms, int N, int M, double tau) {
    double d=std::min((double)N,(double)M); if(d<=0||tau<=0)return 0;
    return ((double)n/d)*(1.0-rms/tau);
}

// --- Umeyama SVD ---
struct SimTransform { double s,theta,tx,ty; bool valid; };
SimTransform umeyama(const double* src, const double* dst, int n) {
    SimTransform r; r.valid=false; r.s=1; r.theta=0; r.tx=0; r.ty=0;
    if(n<2) return r;
    using M2=Eigen::Matrix2d; using V2=Eigen::Vector2d;
    V2 ms=V2::Zero(), md=V2::Zero();
    for(int i=0;i<n;++i){ms+=V2(src[i*2],src[i*2+1]);md+=V2(dst[i*2],dst[i*2+1]);}
    ms/=n; md/=n;
    Eigen::MatrixXd sc(2,n), dc(2,n);
    for(int i=0;i<n;++i){
        sc(0,i)=src[i*2]-ms(0); sc(1,i)=src[i*2+1]-ms(1);
        dc(0,i)=dst[i*2]-md(0); dc(1,i)=dst[i*2+1]-md(1);
    }
    M2 H=sc*dc.transpose();
    Eigen::JacobiSVD<M2> svd(H,Eigen::ComputeFullU|Eigen::ComputeFullV);
    double det=(svd.matrixV().transpose()*svd.matrixU().transpose()).determinant();
    V2 Sv=V2::Ones(); Sv(1)=det;
    M2 R=svd.matrixV()*Sv.asDiagonal()*svd.matrixU().transpose();
    double tr=sc.colwise().squaredNorm().sum();
    if(tr<1e-15) return r;
    double s=svd.singularValues().dot(Sv)/tr;
    if(std::abs(s-1.0)>=0.1) return r;
    double th=std::atan2(R(1,0),R(0,0));
    V2 t=md-s*R*ms;
    r.s=s; r.theta=th; r.tx=t(0); r.ty=t(1); r.valid=true;
    return r;
}

// --- iterative SVD refine (Phase B) ---
struct RefineResult { double s,theta,tx,ty; int n_inliers; double rms; std::vector<int> inlier_mask; bool success; };
RefineResult iterative_svd_refine(const double* U, int N, const double* Wf, int M,
                                   double s, double theta, double tx, double ty,
                                   double s0, int max_iter) {
    RefineResult res; res.s=s; res.theta=theta; res.tx=tx; res.ty=ty;
    res.n_inliers=0; res.rms=1e30; res.success=false;
    if(N<3||M<3) return res;
    std::vector<double> Wt(M*2);
    std::vector<int> prev(N, 0);
    double tau=1.0*s0; apply_similarity(Wf,M,s,theta,tx,ty,Wt.data());
    auto inl=count_inliers_1to1(U,N,Wt.data(),M,tau);
    double sf[]={2.0,5.0,10.0};
    for(int k=0;k<3&&inl.n_inliers<3;++k){tau=sf[k]*s0;inl=count_inliers_1to1(U,N,Wt.data(),M,tau);}
    if(inl.n_inliers<3){res.inlier_mask=std::move(inl.inlier_mask);res.n_inliers=inl.n_inliers;res.rms=inl.rms;return res;}
    prev=inl.inlier_mask;
    for(int iter=0;iter<max_iter;++iter){
        apply_similarity(Wf,M,res.s,res.theta,res.tx,res.ty,Wt.data());
        PointCloud2D cloud; cloud.pts.resize(M);
        for(int i=0;i<M;++i) cloud.pts[i]={Wt[i*2],Wt[i*2+1]};
        KDTree tree(2,cloud,nanoflann::KDTreeSingleIndexAdaptorParams(10));
        struct M2{int u,w;double d;};
        std::vector<M2> cand; cand.reserve(N); double t2=s0*s0;
        for(int i=0;i<N;++i){
            double q[2]={U[i*2],U[i*2+1]}; KDTreeIndexType idx; double ds;
            nanoflann::KNNResultSet<double,KDTreeIndexType> rs(1); rs.init(&idx,&ds);
            tree.findNeighbors(rs,q);
            if(ds<=t2) cand.push_back({i,(int)idx,std::sqrt(ds)});
        }
        std::sort(cand.begin(),cand.end(),[](const M2&a,const M2&b){return a.d<b.d;});
        std::vector<int> wu(M,0); std::vector<double> sp,dp;
        for(auto&c:cand){if(wu[c.w])continue;wu[c.w]=1;
            sp.push_back(Wf[c.w*2]);sp.push_back(Wf[c.w*2+1]);dp.push_back(U[c.u*2]);dp.push_back(U[c.u*2+1]);}
        int np=(int)sp.size()/2; if(np<3) break;
        auto sim=umeyama(sp.data(),dp.data(),np);
        if(!sim.valid||std::abs(sim.s-1.0)>0.1) break;
        res.s=sim.s;res.theta=sim.theta;res.tx=sim.tx;res.ty=sim.ty;
        apply_similarity(Wf,M,res.s,res.theta,res.tx,res.ty,Wt.data());
        inl=count_inliers_1to1(U,N,Wt.data(),M,1.0*s0);
        if(inl.n_inliers<3) break;
        if(inl.inlier_mask==prev) break;
        prev=inl.inlier_mask;
    }
    apply_similarity(Wf,M,res.s,res.theta,res.tx,res.ty,Wt.data());
    inl=count_inliers_1to1(U,N,Wt.data(),M,1.0*s0);
    res.n_inliers=inl.n_inliers;res.rms=inl.rms;res.inlier_mask=std::move(inl.inlier_mask);
    res.success=(res.n_inliers>=3);
    return res;
}

// --- θ_SNR ---
struct ThetaSNRResult { int peak_idx; double peak_deg, snr; };
ThetaSNRResult compute_theta_snr(const double* hist, int nb, double bw) {
    ThetaSNRResult r; r.peak_idx=0; r.peak_deg=0; r.snr=0;
    double pv=0; for(int i=0;i<nb;++i) if(hist[i]>pv){pv=hist[i];r.peak_idx=i;}
    r.peak_deg=(r.peak_idx+0.5)*bw-180.0;
    double bs=0; int bc=0;
    for(int i=0;i<nb;++i) if(std::abs(i-r.peak_idx)>5){bs+=hist[i];bc++;}
    double bm=(bc>10)?bs/bc:1.0; r.snr=pv/std::max(bm,1e-10);
    return r;
}

// ============================================================================
struct PairRecord { int u_idx, w_idx; double theta_deg; int n_in_range_s; };

struct V34PhaseABResult {
    double s, theta, tx, ty; int n_inliers; double rms;
    double peak_snr; int n_samples; std::vector<int> inlier_mask; bool success;
    double theta_peak_deg; int best_n_range; double median_noise;
    int n_phaseb_pairs; int n_phaseb_corr; int n_phasea_records;
    std::vector<PairRecord> records;  // <-- kept for Phase C
    int flip;
};

// ============================================================================
// Phase A+B (V3.3 kept intact, but stores records for Phase C)
// ============================================================================
V34PhaseABResult record_and_filter(
    const double* U, int N, const double* Wf, int M,
    double s0, double s_min, double s_max,
    int K_total, int batch_size, int min_samples,
    int min_inliers, int seed, double fov_diag_asec)
{
    V34PhaseABResult res; res.s=1; res.theta=0; res.tx=0; res.ty=0;
    res.n_inliers=0; res.rms=1e30; res.peak_snr=0; res.n_samples=0; res.success=false;
    res.theta_peak_deg=0; res.best_n_range=0; res.median_noise=0;
    res.n_phaseb_pairs=0; res.n_phaseb_corr=0; res.n_phasea_records=0; res.flip=0;

    if(N<2||M<2) return res;

    std::vector<double> norm_U(N), angle_U(N), norm_Wf(M), angle_Wf(M);
    std::vector<bool> valid_U(N,false), valid_Wf(M,false);
    for(int i=0;i<N;++i){norm_U[i]=std::sqrt(U[i*2]*U[i*2]+U[i*2+1]*U[i*2+1]);angle_U[i]=std::atan2(U[i*2+1],U[i*2]);valid_U[i]=norm_U[i]>1e-10;}
    for(int j=0;j<M;++j){norm_Wf[j]=std::sqrt(Wf[j*2]*Wf[j*2]+Wf[j*2+1]*Wf[j*2+1]);angle_Wf[j]=std::atan2(Wf[j*2+1],Wf[j*2]);valid_Wf[j]=norm_Wf[j]>1e-10;}

    static constexpr int THB=3600; static constexpr double THBW=0.1;
    std::vector<double> th_hist(THB,0);
    std::unordered_set<uint64_t> sampled;
    std::vector<PairRecord> records; records.reserve(K_total);

    std::mt19937 rng(seed);
    std::uniform_int_distribution<int> ud(0,N-1), wd(0,M-1);
    std::vector<double> Wt(M*2);
    double max_t=fov_diag_asec*0.6;
    uint64_t tp=(uint64_t)N*M;
    int Kmax=std::min(K_total,(int)std::min((uint64_t)K_total,tp));

    int n_val=0,best_n=0,last_snr=0;
    for(int iter=0;iter<Kmax;++iter){
        int i=ud(rng),j=wd(rng);
        uint64_t key=(uint64_t)i*M+j; if(sampled.count(key)) continue; sampled.insert(key);
        if(!valid_U[i]||!valid_Wf[j]) continue;
        double s=norm_U[i]/norm_Wf[j]; if(s<s_min||s>s_max) continue;
        double theta=angle_U[i]-angle_Wf[j], td=wrap180(theta*RADTODEG);
        double ct=std::cos(theta),st=std::sin(theta);
        double tx=U[i*2]-s*(ct*Wf[j*2]-st*Wf[j*2+1]),ty=U[i*2+1]-s*(st*Wf[j*2]+ct*Wf[j*2+1]);
        if(std::abs(tx)>max_t||std::abs(ty)>max_t) continue;
        apply_similarity(Wf,M,s,theta,tx,ty,Wt.data());
        int nr=count_s_in_range(U,N,Wt.data(),M,norm_U.data(),norm_Wf.data(),s_min,s_max,5.0*s0);
        int tb=(int)((td+180.0)/THBW); if(tb>=0&&tb<THB) th_hist[tb]+=nr;
        records.push_back({i,j,td,nr}); if(nr>best_n) best_n=nr; n_val++;

        if(n_val>=min_samples&&iter-last_snr>=batch_size){
            last_snr=iter;
            auto snr=compute_theta_snr(th_hist.data(),THB,THBW);
            double t5=std::min(5.0*N,500.0),t10=std::min(10.0*N,1000.0);
            fprintf(stderr,"[vm34] PhaseA: n=%d best=%d peak=%.0f θ=%.2f° SNR=%.1fx (5N=%.0f 10N=%.0f)\n",
                    n_val,best_n,th_hist[snr.peak_idx],snr.peak_deg,snr.snr,t5,t10);
            if(snr.snr>=t10){fprintf(stderr,"[vm34] PhaseA: ≥10N stop\n");break;}
            if(snr.snr>=t5){fprintf(stderr,"[vm34] PhaseA: ≥5N stop\n");break;}
        }
    }
    res.n_samples=n_val; res.records=std::move(records);
    std::vector<int> an; an.reserve(res.records.size());
    for(auto&r:res.records) an.push_back(r.n_in_range_s);
    std::sort(an.begin(),an.end()); double mdn=an[an.size()/2];
    auto fsnr=compute_theta_snr(th_hist.data(),THB,THBW);
    res.peak_snr=fsnr.snr; res.theta_peak_deg=fsnr.peak_deg;
    res.best_n_range=best_n; res.median_noise=mdn; res.n_phasea_records=(int)res.records.size();
    double t5=std::min(5.0*N,500.0);
    fprintf(stderr,"[vm34] PhaseA done: n=%d best=%d med=%.1f pk=%.0f θ=%.2f° SNR=%.1fx (5N=%.0f) rec=%zu\n",
            n_val,best_n,mdn,th_hist[fsnr.peak_idx],fsnr.peak_deg,fsnr.snr,t5,res.records.size());

    // Phase B filter
    double tp_deg=fsnr.peak_deg, nthr=std::max(2.0,1.5*mdn), tband=2.0;
    std::vector<PairRecord> filt;
    for(auto&r:res.records){if(r.n_in_range_s<=(int)nthr)continue;if(angle_diff_deg(r.theta_deg,tp_deg)>tband)continue;filt.push_back(r);}
    if(filt.size()<2){tband=4.0;nthr=std::max(1.0,1.0*mdn);filt.clear();
        for(auto&r:res.records){if(r.n_in_range_s<=(int)nthr)continue;if(angle_diff_deg(r.theta_deg,tp_deg)>tband)continue;filt.push_back(r);}}
    if(filt.size()<2){fprintf(stderr,"[vm34] PhaseB: <2 pairs\n");return res;}
    std::sort(filt.begin(),filt.end(),[](const PairRecord&a,const PairRecord&b){return a.n_in_range_s>b.n_in_range_s;});
    std::vector<int> uu(N,0),wuu(M,0); std::vector<int> cu,cw;
    for(auto&r:filt){if(!uu[r.u_idx]&&!wuu[r.w_idx]){cu.push_back(r.u_idx);cw.push_back(r.w_idx);uu[r.u_idx]=1;wuu[r.w_idx]=1;}}
    fprintf(stderr,"[vm34] PhaseB: %zu corr\n", cu.size());
    if(cu.size()<2){fprintf(stderr,"[vm34] PhaseB: <2 corr\n");return res;}
    std::vector<double> sp(cu.size()*2),dp(cu.size()*2);
    for(size_t k=0;k<cu.size();++k){
        sp[k*2]=Wf[cw[k]*2];sp[k*2+1]=Wf[cw[k]*2+1];dp[k*2]=U[cu[k]*2];dp[k*2+1]=U[cu[k]*2+1];}
    auto sim=umeyama(sp.data(),dp.data(),(int)cu.size());
    if(!sim.valid){fprintf(stderr,"[vm34] PhaseB: SVD invalid\n");return res;}
    apply_similarity(Wf,M,sim.s,sim.theta,sim.tx,sim.ty,Wt.data());
    auto inl=count_inliers_1to1(U,N,Wt.data(),M,1.0*s0);
    auto ref=iterative_svd_refine(U,N,Wf,M,sim.s,sim.theta,sim.tx,sim.ty,s0,10);
    if(ref.success){res.s=ref.s;res.theta=ref.theta;res.tx=ref.tx;res.ty=ref.ty;res.n_inliers=ref.n_inliers;res.rms=ref.rms;res.inlier_mask=std::move(ref.inlier_mask);}
    else{res.s=sim.s;res.theta=sim.theta;res.tx=sim.tx;res.ty=sim.ty;res.n_inliers=inl.n_inliers;res.rms=inl.rms;res.inlier_mask=std::move(inl.inlier_mask);}
    res.success=true;
    res.n_phaseb_pairs=(int)filt.size(); res.n_phaseb_corr=(int)cu.size();
    fprintf(stderr,"[vm34] PhaseB OK: s=%.4f θ=%.2f° n=%d rms=%.3f corr=%zu\n",
            res.s,res.theta*RADTODEG,res.n_inliers,res.rms,cu.size());
    return res;
}

// ============================================================================
// Phase C: 复用Phase A records扩充对应关系
// ============================================================================
struct Pair { int u, w; };

std::vector<Pair> expand_from_records(
    const double* U, int N, const double* Wf, int M, double s0,
    const std::vector<PairRecord>& records,
    double s_sol, double th_sol, double tx_sol, double ty_sol,
    const double* norm_U, const double* norm_Wf)
{
    std::vector<double> Wt(M*2);
    apply_similarity(Wf, M, s_sol, th_sol, tx_sol, ty_sol, Wt.data());

    std::unordered_set<uint64_t> seen;
    std::vector<Pair> candidates;

    // Source 1: dedup records
    for (auto& r : records) {
        uint64_t key = (uint64_t)r.u_idx * M + r.w_idx;
        if (seen.count(key)) continue;
        seen.insert(key);
        double dx = U[r.u_idx*2] - Wt[r.w_idx*2];
        double dy = U[r.u_idx*2+1] - Wt[r.w_idx*2+1];
        if (dx*dx + dy*dy > (5.0*s0)*(5.0*s0)) continue;
        double sr = norm_U[r.u_idx] / norm_Wf[r.w_idx];
        if (sr < 0.9 || sr > 1.1) continue;
        candidates.push_back({r.u_idx, r.w_idx});
    }

    // Source 2: global NN matching of all U→Wt, angle+modulus filter
    PointCloud2D cloud; cloud.pts.resize(M);
    for (int i = 0; i < M; ++i) cloud.pts[i] = {Wt[i*2], Wt[i*2+1]};
    KDTree tree(2, cloud, nanoflann::KDTreeSingleIndexAdaptorParams(10));

    double max_d2 = (5.0*s0)*(5.0*s0);
    for (int k = 0; k < N; ++k) {
        double q[2] = {U[k*2], U[k*2+1]};
        KDTreeIndexType idx; double ds;
        nanoflann::KNNResultSet<double, KDTreeIndexType> rs(1);
        rs.init(&idx, &ds);
        tree.findNeighbors(rs, q);
        if (ds > max_d2) continue;

        double sr = norm_U[k] / norm_Wf[idx];
        if (sr < 0.9 || sr > 1.1) continue;

        uint64_t key = (uint64_t)k * M + idx;
        if (seen.count(key)) continue;
        seen.insert(key);
        candidates.push_back({k, (int)idx});
    }

    fprintf(stderr, "[vm34] PhaseC: %zu unique pairs (sources)\n", candidates.size());

    // 1-to-1 greedy: sort by distance
    std::vector<std::tuple<double,int,int>> scored;
    for (auto& p : candidates) {
        double dx = U[p.u*2] - Wt[p.w*2];
        double dy = U[p.u*2+1] - Wt[p.w*2+1];
        scored.push_back({dx*dx+dy*dy, p.u, p.w});
    }
    std::sort(scored.begin(), scored.end());

    std::vector<int> w_used(M, 0), u_used(N, 0);
    std::vector<Pair> expanded;
    for (auto& [dsq, u, w] : scored) {
        if (w_used[w] || u_used[u]) continue;
        w_used[w] = 1; u_used[u] = 1;
        expanded.push_back({u, w});
    }

    fprintf(stderr, "[vm34] PhaseC: %zu expanded (1to1)\n", expanded.size());
    return expanded;
}

// ============================================================================
// Phase D: 迭代中位数+3σ剔除
// ============================================================================
struct CleanResult {
    std::vector<int> clean_u, clean_w;
    double mad_rms_arcsec;
    int n_removed;
    int iterations;
    double s, theta, tx, ty;
};

CleanResult iterative_mad_clean(
    const double* U, int, const double* Wf, int,
    const std::vector<Pair>& expanded, double s0)
{
    CleanResult cr;
    cr.mad_rms_arcsec = 0;
    cr.n_removed = 0;
    cr.iterations = 0;

    int n = (int)expanded.size();
    if (n < 3) return cr;

    std::vector<bool> keep(n, true);
    SimTransform sim;

    // Initial Umeyama
    {
        std::vector<double> sp, dp;
        for (auto& p : expanded) { sp.push_back(Wf[p.w*2]); sp.push_back(Wf[p.w*2+1]); dp.push_back(U[p.u*2]); dp.push_back(U[p.u*2+1]); }
        sim = umeyama(sp.data(), dp.data(), n);
        if (!sim.valid) return cr;
    }

    int iter = 0;
    int total_removed = 0;
    do {
        int n_removed = 0;

        // Apply transform
        std::vector<double> Wt_vals;
        // We only need Wt for the kept points — compute on the fly
        double ct = std::cos(sim.theta), st = std::sin(sim.theta);

        std::vector<double> dx_list, dy_list;
        std::vector<int> kept_indices;
        for (int i = 0; i < n; ++i) {
            if (!keep[i]) continue;
            auto& p = expanded[i];
            double wx = Wf[p.w*2], wy = Wf[p.w*2+1];
            double wtx = sim.s * (ct*wx - st*wy) + sim.tx;
            double wty = sim.s * (st*wx + ct*wy) + sim.ty;
            dx_list.push_back(U[p.u*2] - wtx);
            dy_list.push_back(U[p.u*2+1] - wty);
            kept_indices.push_back(i);
        }
        int nk = (int)dx_list.size();
        if (nk < 3) break;

        // Median + MAD
        auto dxc = dx_list, dyc = dy_list;
        double mdx = vec_median(dxc), mdy = vec_median(dyc);
        for (auto& v : dxc) v = std::abs(v - mdx);
        for (auto& v : dyc) v = std::abs(v - mdy);
        double sig_x = 1.4826 * vec_median(dxc);
        double sig_y = 1.4826 * vec_median(dyc);
        if (sig_x < 1e-10) sig_x = 1e-10;
        if (sig_y < 1e-10) sig_y = 1e-10;

        // 3σ filter
        for (int j = 0; j < nk; ++j) {
            if (std::abs(dx_list[j] - mdx) > 3.0 * sig_x || std::abs(dy_list[j] - mdy) > 3.0 * sig_y) {
                keep[kept_indices[j]] = false;
                n_removed++;
            }
        }
        total_removed += n_removed;

        if (n_removed > 0) {
            // Re-fit with remaining
            std::vector<double> sp2, dp2;
            for (int i = 0; i < n; ++i) {
                if (!keep[i]) continue;
                auto& p = expanded[i];
                sp2.push_back(Wf[p.w*2]); sp2.push_back(Wf[p.w*2+1]);
                dp2.push_back(U[p.u*2]); dp2.push_back(U[p.u*2+1]);
            }
            sim = umeyama(sp2.data(), dp2.data(), (int)sp2.size()/2);
            if (!sim.valid) break;
        }
        iter++;
    } while (total_removed != (iter == 1 ? total_removed : cr.n_removed) && iter < 10);

    (void)total_removed;
    cr.iterations = iter;
    cr.n_removed = 0;
    for (int i = 0; i < n; ++i) if (!keep[i]) cr.n_removed++;

    // Collect clean pairs
    double ssq = 0;
    int nc = 0;
    double ct = std::cos(sim.theta), st = std::sin(sim.theta);
    for (int i = 0; i < n; ++i) {
        if (!keep[i]) continue;
        auto& p = expanded[i];
        cr.clean_u.push_back(p.u); cr.clean_w.push_back(p.w);

        double wtx = sim.s * (ct*Wf[p.w*2] - st*Wf[p.w*2+1]) + sim.tx;
        double wty = sim.s * (st*Wf[p.w*2] + ct*Wf[p.w*2+1]) + sim.ty;
        double dx = U[p.u*2] - wtx, dy = U[p.u*2+1] - wty;
        ssq += dx*dx + dy*dy; nc++;
    }
    cr.mad_rms_arcsec = (nc > 0) ? std::sqrt(ssq / nc) : 0.0;

    // Final Umeyama with clean pairs only
    if (cr.clean_u.size() >= 2) {
        std::vector<double> sp, dp;
        for (size_t i = 0; i < cr.clean_u.size(); ++i) {
            int w = cr.clean_w[i], u = cr.clean_u[i];
            sp.push_back(Wf[w*2]); sp.push_back(Wf[w*2+1]);
            dp.push_back(U[u*2]); dp.push_back(U[u*2+1]);
        }
        auto final_sim = umeyama(sp.data(), dp.data(), (int)cr.clean_u.size());
        if (final_sim.valid) {
            cr.s = final_sim.s; cr.theta = final_sim.theta;
            cr.tx = final_sim.tx; cr.ty = final_sim.ty;
        } else {
            cr.s = sim.s; cr.theta = sim.theta; cr.tx = sim.tx; cr.ty = sim.ty;
        }
    } else {
        cr.s = sim.s; cr.theta = sim.theta; cr.tx = sim.tx; cr.ty = sim.ty;
    }

    fprintf(stderr, "[vm34] PhaseD: %d removed in %d iters, %zu clean pairs, MAD-RMS=%.3f\"\n",
            cr.n_removed, iter, cr.clean_u.size(), cr.mad_rms_arcsec);
    return cr;
}

// ============================================================================
// Phase E: 6阶SIP多项式拟合
// ============================================================================
struct SIPResult {
    double sip_A[36];
    double sip_B[36];
    double cd[4];
    double crval[2];
    double crpix[2];
    double rms_px;
};

// Returns index of (p,q) term in the 28-term polynomial basis sorted by order
int poly_index(int p, int q) {
    int idx = 0;
    for (int o = 0; o <= 6; ++o)
        for (int pp = 0; pp <= o; ++pp) {
            int qq = o - pp;
            if (pp == p && qq == q) return idx;
            idx++;
        }
    return -1;
}

// Total number of polynomial terms for order <= max_order
int poly_nterms(int max_order) {
    return (max_order + 1) * (max_order + 2) / 2;
}

SIPResult fit_affine_sip(
    const double* U, const double* Wf,
    const std::vector<int>& clean_u, const std::vector<int>& clean_w,
    double s0, double w, double h,
    double center_ra, double center_dec,
    double s_sol, double theta_sol,
    int flip_mode,
    const char* wcs_out_path)
{
    SIPResult sr;
    std::memset(sr.sip_A, 0, sizeof(sr.sip_A));
    std::memset(sr.sip_B, 0, sizeof(sr.sip_B));
    std::memset(sr.cd, 0, sizeof(sr.cd));
    sr.crval[0] = center_ra; sr.crval[1] = center_dec;
    sr.crpix[0] = w / 2.0; sr.crpix[1] = h / 2.0;
    sr.rms_px = 0;

    int M_D = (int)clean_u.size();
    if (M_D < 5) return sr;

    double cx = w / 2.0, cy = h / 2.0;

    // ============================================================
    // 1. CD矩阵: pixel → sky (度/像素)
    //    映射链: pixel(ξ,η) → U(弧秒) → Wf(弧秒) → W(弧秒) → sky(度)
    //    U_x = ξ*s0, U_y = -η*s0  (Y翻转)
    //    Wf = R(-θ)/s · U         (逆相似变换)
    //    W = unflip(Wf)           (flip模式)
    //    Δα = W_x/(3600*cos(δ0)), Δδ = W_y/3600
    // ============================================================
    double ct = std::cos(theta_sol), st = std::sin(theta_sol);
    double cos_dec = std::cos(center_dec * DEGTORAD);
    if (cos_dec < 1e-10) cos_dec = 1e-10;
    bool fx = (flip_mode == 1 || flip_mode == 3);
    bool fy = (flip_mode == 2 || flip_mode == 3);
    double s0_over_s_3600 = s0 / (s_sol * 3600.0);
    double sign_x = fx ? -1.0 : 1.0;
    double sign_y = fy ? -1.0 : 1.0;
    sr.cd[0] = sign_x * s0_over_s_3600 * ct / cos_dec;   // CD1_1
    sr.cd[1] = -sign_x * s0_over_s_3600 * st / cos_dec;  // CD1_2
    sr.cd[2] = -sign_y * s0_over_s_3600 * st;             // CD2_1
    sr.cd[3] = -sign_y * s0_over_s_3600 * ct;             // CD2_2

    // ============================================================
    // 2. 计算每对匹配的SIP修正目标值
    //    WCS-SIP标准:
    //      正向: ξ' = ξ + ΣA_pq·ξ^p·η^q,  [Δα,Δδ] = CD·[ξ',η']
    //      逆推: [ξ',η'] = CD^-1·[Δα,Δδ],  SIP_A = ξ' - ξ
    //
    //    对每对匹配:
    //    a) 从Wf计算天球坐标差: Δα = Wf_x/(3600*cos_dec), Δδ = Wf_y/3600
    //       (Wf是gnomonic投影弧秒坐标，小视场线性近似)
    //    b) 中间坐标: [ξ',η'] = CD^-1 · [Δα,Δδ]
    //    c) 像素偏移: ξ = x_det - CRPIX1, η = y_det - CRPIX2
    //    d) SIP修正目标: sip_dx = ξ' - ξ, sip_dy = η' - η
    // ============================================================
    double cdet = sr.cd[0]*sr.cd[3] - sr.cd[1]*sr.cd[2];
    if (std::abs(cdet) < 1e-30) return sr;
    double cd_inv[4] = { sr.cd[3]/cdet, -sr.cd[1]/cdet, -sr.cd[2]/cdet, sr.cd[0]/cdet };

    std::vector<double> xi_det(M_D), eta_det(M_D);
    std::vector<double> sip_target_x(M_D), sip_target_y(M_D);
    for (int i = 0; i < M_D; ++i) {
        int u_idx = clean_u[i], w_idx = clean_w[i];
        // 检测像素偏移
        double x_det = U[u_idx*2] / s0 + cx;
        double y_det = -U[u_idx*2+1] / s0 + cy;
        xi_det[i] = x_det - sr.crpix[0];
        eta_det[i] = y_det - sr.crpix[1];
        // 天球坐标差（从Wf unflip回W，再转天球坐标）
        double Wf_x = Wf[w_idx*2], Wf_y = Wf[w_idx*2+1];
        double W_x = fx ? -Wf_x : Wf_x;
        double W_y = fy ? -Wf_y : Wf_y;
        double dra = W_x / (3600.0 * cos_dec);
        double ddec = W_y / 3600.0;
        // 中间坐标
        double xi_prime = cd_inv[0]*dra + cd_inv[1]*ddec;
        double eta_prime = cd_inv[2]*dra + cd_inv[3]*ddec;
        // SIP修正目标
        sip_target_x[i] = xi_prime - xi_det[i];
        sip_target_y[i] = eta_prime - eta_det[i];
    }

    // ============================================================
    // 3. 用6阶多项式拟合SIP修正量（仅高阶项p+q>=2）
    //    线性项(p+q<=1)应为0（CD矩阵已覆盖线性映射）
    //    但为了数值稳定性，拟合全部项后只取高阶项
    // ============================================================
    int nterms = poly_nterms(6);  // 28
    Eigen::MatrixXd A(M_D, nterms);
    Eigen::VectorXd bx(M_D), by(M_D);

    for (int i = 0; i < M_D; ++i) {
        bx(i) = sip_target_x[i];
        by(i) = sip_target_y[i];
        // 归一化到[-1,1]提高数值稳定性
        double x = xi_det[i] / (w / 2.0);
        double y = eta_det[i] / (h / 2.0);
        int col = 0;
        for (int o = 0; o <= 6; ++o)
            for (int p = 0; p <= o; ++p) {
                int q = o - p;
                A(i, col++) = std::pow(x, p) * std::pow(y, q);
            }
    }

    Eigen::VectorXd beta_x = (A.transpose() * A).ldlt().solve(A.transpose() * bx);
    Eigen::VectorXd beta_y = (A.transpose() * A).ldlt().solve(A.transpose() * by);

    // ============================================================
    // 4. RMS计算
    // ============================================================
    double full_ssq = 0;
    for (int i = 0; i < M_D; ++i) {
        double pred_x = A.row(i) * beta_x;
        double pred_y = A.row(i) * beta_y;
        full_ssq += (sip_target_x[i] - pred_x) * (sip_target_x[i] - pred_x)
                  + (sip_target_y[i] - pred_y) * (sip_target_y[i] - pred_y);
    }
    sr.rms_px = std::sqrt(full_ssq / M_D);

    // ============================================================
    // 5. 提取SIP系数(仅高阶项p+q>=2)
    //    归一化空间: x_n = ξ/(w/2), y_n = η/(h/2)
    //    β·x_n^p·y_n^q = β·ξ^p·η^q / ((w/2)^p·(h/2)^q)
    //    SIP标准: A_pq = β / ((w/2)^p·(h/2)^q)
    // ============================================================
    for (int order = 2; order <= 6; ++order)
        for (int p = 0; p <= order; ++p) {
            int q = order - p;
            int idx = poly_index(p, q);
            if (idx < 0) continue;
            double norm_factor = std::pow(w / 2.0, p) * std::pow(h / 2.0, q);
            sr.sip_A[p * 6 + q] = beta_x(idx) / norm_factor;
            sr.sip_B[p * 6 + q] = beta_y(idx) / norm_factor;
        }

    // 调试: 打印前10对的SIP修正量
    fprintf(stderr, "[vm34] PhaseE SIP: %d pairs, RMS=%.6fpx\n", M_D, sr.rms_px);
    fprintf(stderr, "[vm34] CD=[[%.6e,%.6e],[%.6e,%.6e]] CRVAL=[%.10f,%.10f] CRPIX=[%.1f,%.1f]\n",
            sr.cd[0], sr.cd[1], sr.cd[2], sr.cd[3], sr.crval[0], sr.crval[1], sr.crpix[0], sr.crpix[1]);
    int n_dbg = std::min(10, M_D);
    fprintf(stderr, "[vm34] SIP target (first %d): ξ_det  η_det  →  sip_dx  sip_dy\n", n_dbg);
    for (int i = 0; i < n_dbg; ++i) {
        fprintf(stderr, "[vm34]   %8.1f %8.1f  →  %+8.4f %+8.4f\n",
                xi_det[i], eta_det[i], sip_target_x[i], sip_target_y[i]);
    }

    // 6. 输出JSON格式WCS文件（绕开ctypes对齐问题）
    if (wcs_out_path && wcs_out_path[0]) {
        FILE* fw = fopen(wcs_out_path, "w");
        if (fw) {
            fprintf(fw, "{\n");
            fprintf(fw, "  \"CD\": [[%.16e, %.16e], [%.16e, %.16e]],\n",
                    sr.cd[0], sr.cd[1], sr.cd[2], sr.cd[3]);
            fprintf(fw, "  \"CRVAL\": [%.10f, %.10f],\n", sr.crval[0], sr.crval[1]);
            fprintf(fw, "  \"CRPIX\": [%.3f, %.3f],\n", sr.crpix[0], sr.crpix[1]);
            fprintf(fw, "  \"SIP_A\": [");
            for (int i = 0; i < 36; ++i) fprintf(fw, "%s%.16e", i?",":"", sr.sip_A[i]);
            fprintf(fw, "],\n  \"SIP_B\": [");
            for (int i = 0; i < 36; ++i) fprintf(fw, "%s%.16e", i?",":"", sr.sip_B[i]);
            fprintf(fw, "],\n  \"RMS_PX\": %.6f\n", sr.rms_px);
            fprintf(fw, "}\n");
            fclose(fw);
            fprintf(stderr, "[vm34] WCS JSON written to: %s\n", wcs_out_path);
        } else {
            fprintf(stderr, "[vm34] ERROR: cannot open WCS output: %s\n", wcs_out_path);
        }
    }

    return sr;
}

// ============================================================================
// Mode result
// ============================================================================
struct ModeRes {
    bool success; double norm_score; double peak_snr; int n_samples;
    double s, theta, tx, ty; int n_inliers; double rms; int mode;
    std::vector<int> inlier_mask;
    V34PhaseABResult ab;
};

ModeRes solve_single_mode(const double* U, int N, const double* W, int M,
                           int mode, double s0, double s_min, double s_max,
                           int K_total, int batch_size, int min_samples,
                           int min_inliers, int seed, double fov_diag_asec,
                           volatile std::atomic<bool>* early_exit)
{
    ModeRes mr; mr.success=false; mr.norm_score=0; mr.peak_snr=0; mr.n_samples=0; mr.mode=mode;
    if(early_exit && early_exit->load(std::memory_order_relaxed)){fprintf(stderr,"[vm34] mode%d: skip\n",mode);return mr;}
    std::vector<double> Wf(M*2); apply_flip(W,M,mode,Wf.data());
    fprintf(stderr,"[vm34] mode%d: M=%d\n",mode,M);
    mr.ab=record_and_filter(U,N,Wf.data(),M,s0,s_min,s_max,K_total,batch_size,min_samples,min_inliers,seed+mode,fov_diag_asec);
    mr.peak_snr=mr.ab.peak_snr; mr.n_samples=mr.ab.n_samples;
    if(!mr.ab.success){fprintf(stderr,"[vm34] mode%d: fail\n",mode);return mr;}
    mr.s=mr.ab.s; mr.theta=mr.ab.theta; mr.tx=mr.ab.tx; mr.ty=mr.ab.ty;
    mr.n_inliers=mr.ab.n_inliers; mr.rms=mr.ab.rms;
    mr.inlier_mask=std::move(mr.ab.inlier_mask);
    mr.norm_score=compute_normalized_score(mr.n_inliers,mr.rms,N,M,1.0*s0);
    mr.success=true;
    if(mr.success&&mr.s>=0.9&&mr.s<=1.1&&early_exit){early_exit->store(true,std::memory_order_relaxed);fprintf(stderr,"[vm34] mode%d converged\n",mode);}
    fprintf(stderr,"[vm34] mode%d final: s=%.4f θ=%.2f° n=%d rms=%.3f norm=%.4f\n",mode,mr.s,mr.theta*RADTODEG,mr.n_inliers,mr.rms,mr.norm_score);
    return mr;
}

} // namespace vm34

// ============================================================================
// vm34_solve
// ============================================================================
extern "C" VM34_API int vm34_solve(
    const double* U, int N_img, const double* W, int M,
    const VM34SolveParams* params, VM34SolveResult* result)
{
    using namespace vm34;
    int* saved_mask = result->inlier_mask;
    std::memset(result, 0, sizeof(VM34SolveResult));
    result->inlier_mask = saved_mask;
    result->norm_score = -1.0; result->rms = 1e30;

    if (N_img < 2 || M < 2) { fprintf(stderr, "[vm34] N=%d M=%d too few\n", N_img, M); return -1; }

    int n_modes = std::max(1, std::min(params->n_modes, 4));
    std::vector<ModeRes> mres(n_modes);
    std::atomic<bool> early_exit(false);

    #pragma omp parallel for schedule(static)
    for (int mode = 0; mode < n_modes; ++mode) {
        mres[mode] = solve_single_mode(U, N_img, W, M, mode, params->s0,
            params->s_min, params->s_max, params->K_total, params->batch_size,
            params->min_samples, params->min_inliers, params->seed,
            params->fov_diag_asec, &early_exit);
    }

    double best_score = -1.0; int best_mode = -1;
    for (int m = 0; m < n_modes; ++m)
        if (mres[m].success && mres[m].norm_score > best_score)
            { best_score = mres[m].norm_score; best_mode = m; }

    if (best_mode < 0) { fprintf(stderr, "[vm34] all fail (best=%.4f)\n", best_score); return -1; }

    auto& best = mres[best_mode];

    // Phase C: expand
    std::vector<double> Wf(M*2);
    apply_flip(W, M, best_mode, Wf.data());

    // Precompute norms for Phase C
    std::vector<double> norm_U(N_img), norm_Wf(M);
    for (int i = 0; i < N_img; ++i) norm_U[i] = std::sqrt(U[i*2]*U[i*2] + U[i*2+1]*U[i*2+1]);
    for (int j = 0; j < M; ++j) norm_Wf[j] = std::sqrt(Wf[j*2]*Wf[j*2] + Wf[j*2+1]*Wf[j*2+1]);

    auto expanded = expand_from_records(U, N_img, Wf.data(), M, params->s0,
        best.ab.records, best.s, best.theta, best.tx, best.ty,
        norm_U.data(), norm_Wf.data());

    if (expanded.size() < 3) {
        // Fallback: use Phase B result
        result->s = best.s; result->theta = best.theta;
        result->tx = best.tx; result->ty = best.ty;
        result->n_inliers = best.n_inliers; result->rms = best.rms;
        result->best_mode = best_mode; result->norm_score = best.norm_score;
        result->peak_snr = best.peak_snr; result->n_samples = best.n_samples;
        result->success = 1;
        for (int i = 0; i < N_img; ++i) result->inlier_mask[i] = best.inlier_mask[i];
        // 设置线性CD + 零SIP (pixel→sky, 含flip和cos(δ))
        double ct0 = std::cos(best.theta), st0 = std::sin(best.theta);
        double cos_dfb = std::cos(params->center_dec * DEGTORAD);
        if (cos_dfb < 1e-10) cos_dfb = 1e-10;
        double s0_s_3600 = params->s0 / (best.s * 3600.0);
        bool fx0 = (best.mode == 1 || best.mode == 3);
        bool fy0 = (best.mode == 2 || best.mode == 3);
        double sx0 = fx0 ? -1.0 : 1.0, sy0 = fy0 ? -1.0 : 1.0;
        result->cd[0] = sx0 * s0_s_3600 * ct0 / cos_dfb;
        result->cd[1] = -sx0 * s0_s_3600 * st0 / cos_dfb;
        result->cd[2] = -sy0 * s0_s_3600 * st0;
        result->cd[3] = -sy0 * s0_s_3600 * ct0;
        result->crval[0] = params->center_ra - best.tx / (std::max(cos_dfb, 1e-10) * 3600.0);
        result->crval[1] = params->center_dec - best.ty / 3600.0;
        result->crpix[0] = params->img_width/2; result->crpix[1] = params->img_height/2;
        for (int i = 0; i < 36; ++i) { result->sip_A[i] = 0; result->sip_B[i] = 0; }
        fprintf(stderr, "[vm34] PhaseC: expanded=%zu (<3), fallback to AB\n", expanded.size());
        return 0;
    }

    // Phase D: iterative MAD clean
    auto clean = iterative_mad_clean(U, N_img, Wf.data(), M, expanded, params->s0);
    if (clean.clean_u.size() < 3) {
        result->s = best.s; result->theta = best.theta;
        result->tx = best.tx; result->ty = best.ty;
        result->n_inliers = best.n_inliers; result->rms = best.rms;
        result->best_mode = best_mode; result->norm_score = best.norm_score;
        result->peak_snr = best.peak_snr; result->n_samples = best.n_samples;
        result->success = 1;
        for (int i = 0; i < N_img; ++i) result->inlier_mask[i] = best.inlier_mask[i];
        // 设置线性CD + 零SIP (pixel→sky, 含flip和cos(δ))
        double ct1 = std::cos(best.theta), st1 = std::sin(best.theta);
        double cos_dfb2 = std::cos(params->center_dec * DEGTORAD);
        if (cos_dfb2 < 1e-10) cos_dfb2 = 1e-10;
        double s0_s_3600b = params->s0 / (best.s * 3600.0);
        bool fx1 = (best.mode == 1 || best.mode == 3);
        bool fy1 = (best.mode == 2 || best.mode == 3);
        double sx1 = fx1 ? -1.0 : 1.0, sy1 = fy1 ? -1.0 : 1.0;
        result->cd[0] = sx1 * s0_s_3600b * ct1 / cos_dfb2;
        result->cd[1] = -sx1 * s0_s_3600b * st1 / cos_dfb2;
        result->cd[2] = -sy1 * s0_s_3600b * st1;
        result->cd[3] = -sy1 * s0_s_3600b * ct1;
        result->crval[0] = params->center_ra - best.tx / (std::max(cos_dfb2, 1e-10) * 3600.0);
        result->crval[1] = params->center_dec - best.ty / 3600.0;
        result->crpix[0] = params->img_width/2; result->crpix[1] = params->img_height/2;
        for (int i = 0; i < 36; ++i) { result->sip_A[i] = 0; result->sip_B[i] = 0; }
        fprintf(stderr, "[vm34] PhaseD: clean=%zu (<3), fallback to AB\n", clean.clean_u.size());
        return 0;
    }

    // Phase E: SIP 多项式拟合（用Phase D干净对应关系）
    // 计算精修后的天球中心: 原始中心 + 平移偏移
    double cos_d0 = std::cos(params->center_dec * DEGTORAD);
    double refined_ra = params->center_ra - clean.tx / (std::max(cos_d0, 1e-10) * 3600.0);
    double refined_dec = params->center_dec - clean.ty / 3600.0;
    auto sip = fit_affine_sip(U, Wf.data(), clean.clean_u, clean.clean_w,
        params->s0, params->img_width, params->img_height,
        refined_ra, refined_dec, clean.s, clean.theta, best.mode, params->wcs_out_path);

    // 存储SIP结果
    for (int i = 0; i < 36; ++i) {
        result->sip_A[i] = sip.sip_A[i];
        result->sip_B[i] = sip.sip_B[i];
    }
    result->cd[0] = sip.cd[0]; result->cd[1] = sip.cd[1];
    result->cd[2] = sip.cd[2]; result->cd[3] = sip.cd[3];
    result->crval[0] = sip.crval[0]; result->crval[1] = sip.crval[1];
    result->crpix[0] = sip.crpix[0]; result->crpix[1] = sip.crpix[1];

    // Output: use Phase D refined transform for s/theta/tx/ty
    result->s = clean.s; result->theta = clean.theta;
    result->tx = clean.tx; result->ty = clean.ty;
    result->n_inliers = (int)clean.clean_u.size();
    result->rms = clean.mad_rms_arcsec / params->s0;  // px
    result->best_mode = best_mode;
    result->norm_score = best.norm_score;
    result->peak_snr = best.peak_snr;
    result->n_samples = best.n_samples;
    result->success = 1;
    for (int i = 0; i < N_img; ++i) result->inlier_mask[i] = best.inlier_mask[i];

    // Debug
    result->debug.theta_snr = best.ab.peak_snr;
    result->debug.theta_peak_deg = best.ab.theta_peak_deg;
    result->debug.best_n_range = best.ab.best_n_range;
    result->debug.median_noise = best.ab.median_noise;
    result->debug.n_phaseb_pairs = best.ab.n_phaseb_pairs;
    result->debug.n_phaseb_corr = best.ab.n_phaseb_corr;
    result->debug.n_phasea_records = best.ab.n_phasea_records;
    result->debug.n_phasec_expanded = (int)expanded.size();
    result->debug.n_phased_clean = (int)clean.clean_u.size();
    result->debug.n_phased_iterations = clean.iterations;
    result->debug.mad_rms_arcsec = clean.mad_rms_arcsec;

    fprintf(stderr, "[vm34] OK: mode=%d s=%.6f θ=%.2f° tx=%.3f\" ty=%.3f\" n=%d SNR=%.1fx "
            "SIP-RMS=%.3fpx C-exp=%zu D-clean=%zu D-iter=%d\n",
            best_mode, clean.s, clean.theta*RADTODEG, clean.tx, clean.ty,
            (int)clean.clean_u.size(), best.peak_snr,
            sip.rms_px, expanded.size(), clean.clean_u.size(), clean.iterations);
    return 0;
}

extern "C" VM34_API int vm34_count_inliers(
    const double* U, int N_img, const double* W, int M,
    double s, double theta, double tx, double ty,
    double s0, int* inlier_mask, double* out_rms)
{
    using namespace vm34;
    std::vector<double> Wt(M*2);
    apply_similarity(W, M, s, theta, tx, ty, Wt.data());
    auto inl = count_inliers_1to1(U, N_img, Wt.data(), M, 1.0*s0);
    for (int i = 0; i < N_img; ++i) inlier_mask[i] = inl.inlier_mask[i];
    *out_rms = inl.rms;
    return inl.n_inliers;
}
