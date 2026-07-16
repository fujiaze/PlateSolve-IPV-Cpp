/**
 * vm35_core.cpp - V3.5 精简版: Phase A+B SNR验证 → 直送分层拟合
 *
 * Phase A: 1点SNR抽样 + 稀疏度加权 + θ直方图 + 5N/10N停止
 * Phase B: 三级放宽过滤 → 1对1互斥 → cu/cw匹配对 + SVD
 * Phase C: 分层拟合 — Layer0(Umeyama→CD+符号验正) → Layer1(MAD+全仿射→更新CD/CRVAL) → Layer2(BIC SIP)
 *
 * V3.5最终版(2026-06-09): 移除全局NN扩充(C)、迭代MAD清洗(D)、星点扩增(D')
 * 根因: 全局NN和扩增在变换不完全精确时引入大量假匹配对，导致拟合精度恶化
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

#include "../include/vm35_api.h"
#include "Eigen/Dense"
#include "nanoflann.hpp"

namespace vm35 {

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

// --- 基础几何 ---
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

// --- 中位数 ---
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

// --- 迭代SVD精修 (Phase B) ---
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

struct V35PhaseABResult {
    double s, theta, tx, ty; int n_inliers; double rms;
    double peak_snr; int n_samples; std::vector<int> inlier_mask; bool success;
    double theta_peak_deg; int best_n_range; double median_noise;
    int n_phaseb_pairs; int n_phaseb_corr; int n_phasea_records;
    std::vector<PairRecord> records;
    std::vector<int> cu, cw; // Phase B验证匹配对的U和Wf索引
    int flip;
};

// ============================================================================
// Phase A+B (N=250 + 稀疏度加权抽样 + 5N/10N停止)
// ============================================================================
V35PhaseABResult record_and_filter(
    const double* U, int N, const double* Wf, int M,
    double s0, double s_min, double s_max,
    int K_total, int batch_size, int min_samples,
    int min_inliers, int seed, double fov_diag_asec)
{
    V35PhaseABResult res; res.s=1; res.theta=0; res.tx=0; res.ty=0;
    res.n_inliers=0; res.rms=1e30; res.peak_snr=0; res.n_samples=0; res.success=false;
    res.theta_peak_deg=0; res.best_n_range=0; res.median_noise=0;
    res.n_phaseb_pairs=0; res.n_phaseb_corr=0; res.n_phasea_records=0; res.flip=0;

    if(N<2||M<2) return res;

    std::vector<double> norm_U(N), angle_U(N), norm_Wf(M), angle_Wf(M);
    std::vector<bool> valid_U(N,false), valid_Wf(M,false);
    for(int i=0;i<N;++i){norm_U[i]=std::sqrt(U[i*2]*U[i*2]+U[i*2+1]*U[i*2+1]);angle_U[i]=std::atan2(U[i*2+1],U[i*2]);valid_U[i]=norm_U[i]>1e-10;}
    for(int j=0;j<M;++j){norm_Wf[j]=std::sqrt(Wf[j*2]*Wf[j*2]+Wf[j*2+1]*Wf[j*2+1]);angle_Wf[j]=std::atan2(Wf[j*2+1],Wf[j*2]);valid_Wf[j]=norm_Wf[j]>1e-10;}

    // 稀疏度计算: 第3近邻距离(0=自身,1=最近邻,2,3) → 局部星点密度反比
    auto compute_sparsity3 = [](const double* pts, int NP) -> std::vector<double> {
        PointCloud2D cloud; cloud.pts.resize(NP);
        for(int i=0;i<NP;++i) cloud.pts[i]={pts[i*2],pts[i*2+1]};
        KDTree tree(2,cloud,nanoflann::KDTreeSingleIndexAdaptorParams(10));
        tree.buildIndex();
        std::vector<double> sp(NP);
        int kk = std::min(3, NP-1);
        for(int i=0;i<NP;++i){
            double q[2]={pts[i*2],pts[i*2+1]};
            std::vector<KDTreeIndexType> idx(kk+1);
            std::vector<double> dists(kk+1);
            nanoflann::KNNResultSet<double,KDTreeIndexType> rs(kk+1);
            rs.init(idx.data(),dists.data());
            tree.findNeighbors(rs,q);
            sp[i]=std::sqrt(dists[kk]);
        }
        return sp;
    };
    auto sparsity_U = compute_sparsity3(U,N);
    auto sparsity_W = compute_sparsity3(Wf,M);

    // 稀疏度排序: 稀疏→密集 (sparsity值越小=密度越高=星越密集)
    std::vector<int> u_by_sparsity(N), w_by_sparsity(M);
    std::iota(u_by_sparsity.begin(),u_by_sparsity.end(),0);
    std::iota(w_by_sparsity.begin(),w_by_sparsity.end(),0);
    std::sort(u_by_sparsity.begin(),u_by_sparsity.end(),
        [&](int a,int b){return sparsity_U[a]<sparsity_U[b];});
    std::sort(w_by_sparsity.begin(),w_by_sparsity.end(),
        [&](int a,int b){return sparsity_W[a]<sparsity_W[b];});
    // 稀疏度排名(0=最密集, N-1=最稀疏)
    std::vector<int> u_rank(N), w_rank(M);
    for(int i=0;i<N;++i) u_rank[u_by_sparsity[i]]=i;
    for(int j=0;j<M;++j) w_rank[w_by_sparsity[j]]=j;

    double sp_med_u = sparsity_U[u_by_sparsity[N/2]];
    double sp_med_w = sparsity_W[w_by_sparsity[M/2]];
    fprintf(stderr,"[vm35] PhaseA 稀疏度: U中位=%.1f\" W中位=%.1f\"\n",sp_med_u,sp_med_w);

    static constexpr int THB=3600; static constexpr double THBW=0.1;
    std::vector<double> th_hist(THB,0);
    std::unordered_set<uint64_t> sampled;
    std::vector<PairRecord> records; records.reserve(K_total);

    std::mt19937 rng(seed);
    std::uniform_int_distribution<int> ud(0,N-1), wd(0,M-1);
    std::uniform_real_distribution<double> u01(0.0,1.0);
    std::normal_distribution<double> ngauss(0.0,1.0);
    std::vector<double> Wt(M*2);
    double max_t=fov_diag_asec*0.6;
    uint64_t tp=(uint64_t)N*M;
    int Kmax=std::min(K_total,(int)std::min((uint64_t)K_total,tp));
    double p_weighted=0.7; // 70%稀疏度加权, 30%均匀随机

    int n_val=0,best_n=0,last_snr=0;
    for(int iter=0;iter<Kmax;++iter){
        int i,j;
        if(u01(rng) < p_weighted){
            // 稀疏度加权: 选相似密度的星对
            // 随机选U的rank_r, 选W中rank相近的星(加高斯噪声)
            int rank_r = std::uniform_int_distribution<int>(0,std::min(N,M)-1)(rng);
            int sigma_u = std::max(1,(int)(N*0.12));
            int sigma_w = std::max(1,(int)(M*0.12));
            int ri = rank_r + (int)(ngauss(rng)*sigma_u);
            int rj = rank_r + (int)(ngauss(rng)*sigma_w);
            ri=std::max(0,std::min(N-1,ri));
            rj=std::max(0,std::min(M-1,rj));
            i=u_by_sparsity[ri]; j=w_by_sparsity[rj];
        } else {
            i=ud(rng); j=wd(rng);
        }
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
        records.push_back({i,j,td,nr});
        if(nr>best_n){
            best_n=nr;
            fprintf(stderr,"[vm35] PhaseA: new best_n=%d (i=%d j=%d s=%.4f θ=%.2f° tx=%.1f\" ty=%.1f\")\n",
                    nr,i,j,s,td,tx,ty);
        }
        n_val++;

        // 回退V3.3: 只用5N/10N动态阈值停止
        if(n_val>=min_samples&&iter-last_snr>=batch_size){
            last_snr=iter;
            auto snr=compute_theta_snr(th_hist.data(),THB,THBW);
            double t5=std::min(5.0*N,500.0),t10=std::min(10.0*N,1000.0);
            fprintf(stderr,"[vm35] PhaseA: n=%d best=%d peak=%.0f θ=%.2f° SNR=%.1fx (5N=%.0f 10N=%.0f)\n",
                    n_val,best_n,th_hist[snr.peak_idx],snr.peak_deg,snr.snr,t5,t10);
            if(snr.snr>=t10){fprintf(stderr,"[vm35] PhaseA: ≥10N stop\n");break;}
            if(snr.snr>=t5){fprintf(stderr,"[vm35] PhaseA: ≥5N stop\n");break;}
        }
    }
    res.n_samples=n_val; res.records=std::move(records);
    std::vector<int> an; an.reserve(res.records.size());
    for(auto&r:res.records) an.push_back(r.n_in_range_s);
    std::sort(an.begin(),an.end()); double mdn=an[an.size()/2];
    auto fsnr=compute_theta_snr(th_hist.data(),THB,THBW);
    res.peak_snr=fsnr.snr; res.theta_peak_deg=fsnr.peak_deg;
    res.best_n_range=best_n; res.median_noise=mdn;

    res.n_phasea_records=(int)res.records.size();
    double t5=std::min(5.0*N,500.0);
    fprintf(stderr,"[vm35] PhaseA done: n=%d best=%d med=%.1f pk=%.0f θ=%.2f° SNR=%.1fx (5N=%.0f) rec=%zu\n",
            n_val,best_n,mdn,th_hist[fsnr.peak_idx],fsnr.peak_deg,fsnr.snr,t5,res.records.size());

    // 调试: θ直方图top-5峰值
    {
        std::vector<std::pair<double,int>> hist_bins(THB);
        for(int b=0;b<THB;++b) hist_bins[b]={th_hist[b],b};
        std::partial_sort(hist_bins.begin(), hist_bins.begin()+std::min(5,THB), hist_bins.end(),
            [](const auto&a,const auto&b){return a.first>b.first;});
        fprintf(stderr,"[vm35] PhaseA θ直方图top5:");
        for(int k=0;k<std::min(5,THB);++k){
            double deg=(hist_bins[k].second*THBW-180.0);
            fprintf(stderr," [%.1f°:%.0f]",deg,hist_bins[k].first);
        }
        fprintf(stderr,"\n");
    }

    // 调试: n_in_range分布 (P25/P50/P75/P90/P95/P99/max)
    {
        std::vector<int> nr_sorted; nr_sorted.reserve(res.records.size());
        for(auto&r:res.records) nr_sorted.push_back(r.n_in_range_s);
        std::sort(nr_sorted.begin(),nr_sorted.end());
        int nrs=(int)nr_sorted.size();
        auto pct=[&](double p)->int{ return nr_sorted[std::min((int)(p*nrs/100.0),nrs-1)]; };
        fprintf(stderr,"[vm35] PhaseA n_in_range分布: min=%d P25=%d P50=%d P75=%d P90=%d P95=%d P99=%d max=%d\n",
                nr_sorted.empty()?0:nr_sorted[0], pct(25), pct(50), pct(75), pct(90), pct(95), pct(99),
                nr_sorted.empty()?0:nr_sorted.back());
    }

    // 调试: θ峰值±2°内的records统计
    {
        int n_in_band=0, n_high=0;
        double sum_nr=0;
        for(auto&r:res.records){
            if(angle_diff_deg(r.theta_deg,fsnr.peak_deg)<=2.0){
                n_in_band++;
                sum_nr+=r.n_in_range_s;
                if(r.n_in_range_s>=3) n_high++;
            }
        }
        fprintf(stderr,"[vm35] PhaseA θ峰值±2°: %d records (n≥3: %d), 平均n_in_range=%.1f\n",
                n_in_band, n_high, n_in_band>0?sum_nr/n_in_band:0.0);
    }

    // 调试: 对best_n最高的record做详细变换分析
    {
        const PairRecord* best_rec=nullptr;
        for(auto&r:res.records) if(r.n_in_range_s==best_n){best_rec=&r;break;}
        if(best_rec){
            int bi=best_rec->u_idx, bj=best_rec->w_idx;
            double s=norm_U[bi]/norm_Wf[bj];
            double theta=angle_U[bi]-angle_Wf[bj];
            double ct=std::cos(theta),st=std::sin(theta);
            double tx=U[bi*2]-s*(ct*Wf[bj*2]-st*Wf[bj*2+1]);
            double ty=U[bi*2+1]-s*(st*Wf[bj*2]+ct*Wf[bj*2+1]);
            std::vector<double> Wt(M*2); apply_similarity(Wf,M,s,theta,tx,ty,Wt.data());
            // 统计每颗图像星到最近邻Gaia星的距离
            PointCloud2D cloud; cloud.pts.resize(M);
            for(int k=0;k<M;++k) cloud.pts[k]={Wt[k*2],Wt[k*2+1]};
            KDTree tree(2,cloud,nanoflann::KDTreeSingleIndexAdaptorParams(10));
            double d2max=5.0*s0*5.0*s0;
            int n_match=0, n_dist_ok=0, n_scale_ok=0;
            std::vector<double> match_dists;
            for(int k=0;k<N;++k){
                double q[2]={U[k*2],U[k*2+1]}; KDTreeIndexType idx; double ds;
                nanoflann::KNNResultSet<double,KDTreeIndexType> rs(1); rs.init(&idx,&ds);
                tree.findNeighbors(rs,q);
                double dist=std::sqrt(ds);
                double sr=norm_U[k]/norm_Wf[idx];
                bool dist_ok=(ds<=d2max), scale_ok=(sr>=s_min&&sr<=s_max);
                if(dist_ok) n_dist_ok++;
                if(scale_ok) n_scale_ok++;
                if(dist_ok&&scale_ok){n_match++; match_dists.push_back(dist);}
            }
            fprintf(stderr,"[vm35] PhaseA best_record分析: i=%d j=%d s=%.4f θ=%.2f°\n",bi,bj,s,best_rec->theta_deg);
            fprintf(stderr,"[vm35]   N=%d M=%d max_dist=%.1f\" s_range=[%.2f,%.2f]\n",N,M,5.0*s0,s_min,s_max);
            fprintf(stderr,"[vm35]   距离OK: %d/%d  scaleOK: %d/%d  两者OK: %d/%d\n",
                    n_dist_ok,N,n_scale_ok,N,n_match,N);
            if(!match_dists.empty()){
                std::sort(match_dists.begin(),match_dists.end());
                fprintf(stderr,"[vm35]   匹配距离: min=%.2f\" P50=%.2f\" max=%.2f\"\n",
                        match_dists.front(),match_dists[match_dists.size()/2],match_dists.back());
            }
        }
    }

    // Phase B filter
    // 自适应阈值：records少时放宽条件
    double tp_deg=fsnr.peak_deg;
    int n_rec = (int)res.records.size();
    double nthr, tband;
    std::vector<PairRecord> filt;

    if (n_rec >= 10) {
        // 正常情况：严格过滤
        nthr = std::max(2.0, 1.5*mdn); tband = 2.0;
        for(auto&r:res.records){if(r.n_in_range_s<=(int)nthr)continue;if(angle_diff_deg(r.theta_deg,tp_deg)>tband)continue;filt.push_back(r);}
    }
    if (filt.size() < 2) {
        // 放宽：降低n_in_range阈值，扩大θ带宽
        nthr = std::max(1.0, 1.0*mdn); tband = 4.0;
        filt.clear();
        for(auto&r:res.records){if(r.n_in_range_s<(int)nthr)continue;if(angle_diff_deg(r.theta_deg,tp_deg)>tband)continue;filt.push_back(r);}
    }
    if (filt.size() < 2) {
        // 最终放宽：n_in_range≥1，θ带宽8°
        nthr = 1; tband = 8.0;
        filt.clear();
        for(auto&r:res.records){if(r.n_in_range_s<(int)nthr)continue;if(angle_diff_deg(r.theta_deg,tp_deg)>tband)continue;filt.push_back(r);}
    }
    if(filt.size()<2){fprintf(stderr,"[vm35] PhaseB: <2 pairs\n");return res;}
    std::sort(filt.begin(),filt.end(),[](const PairRecord&a,const PairRecord&b){return a.n_in_range_s>b.n_in_range_s;});
    std::vector<int> uu(N,0),wuu(M,0); std::vector<int> cu,cw;
    for(auto&r:filt){if(!uu[r.u_idx]&&!wuu[r.w_idx]){cu.push_back(r.u_idx);cw.push_back(r.w_idx);uu[r.u_idx]=1;wuu[r.w_idx]=1;}}
    res.cu = cu; res.cw = cw;
    fprintf(stderr,"[vm35] PhaseB: %zu corr\n", cu.size());
    if(cu.size()<2){fprintf(stderr,"[vm35] PhaseB: <2 corr\n");return res;}
    std::vector<double> sp(cu.size()*2),dp(cu.size()*2);
    for(size_t k=0;k<cu.size();++k){
        sp[k*2]=Wf[cw[k]*2];sp[k*2+1]=Wf[cw[k]*2+1];dp[k*2]=U[cu[k]*2];dp[k*2+1]=U[cu[k]*2+1];}
    auto sim=umeyama(sp.data(),dp.data(),(int)cu.size());
    if(!sim.valid){fprintf(stderr,"[vm35] PhaseB: SVD invalid\n");return res;}
    apply_similarity(Wf,M,sim.s,sim.theta,sim.tx,sim.ty,Wt.data());
    auto inl=count_inliers_1to1(U,N,Wt.data(),M,1.0*s0);
    // 调试: SVD变换后的匹配距离分布
    {
        PointCloud2D cloud2; cloud2.pts.resize(M);
        for(int k=0;k<M;++k) cloud2.pts[k]={Wt[k*2],Wt[k*2+1]};
        KDTree tree2(2,cloud2,nanoflann::KDTreeSingleIndexAdaptorParams(10));
        std::vector<double> dists;
        for(int k=0;k<N;++k){
            double q[2]={U[k*2],U[k*2+1]}; KDTreeIndexType idx2; double ds2;
            nanoflann::KNNResultSet<double,KDTreeIndexType> rs2(1); rs2.init(&idx2,&ds2);
            tree2.findNeighbors(rs2,q);
            double d=std::sqrt(ds2);
            if(d<5.0*s0) dists.push_back(d);
        }
        if(!dists.empty()){
            std::sort(dists.begin(),dists.end());
            fprintf(stderr,"[vm35] PhaseB SVD后距离: n_near=%zu min=%.2f\" P25=%.2f\" P50=%.2f\" P75=%.2f\" max=%.2f\" (tau=%.2f\")\n",
                    dists.size(),dists.front(),dists[dists.size()/4],dists[dists.size()/2],
                    dists[3*dists.size()/4],dists.back(),1.0*s0);
        } else {
            fprintf(stderr,"[vm35] PhaseB SVD后: 无近邻 (5*s0=%.1f\")\n",5.0*s0);
        }
    }
    auto ref=iterative_svd_refine(U,N,Wf,M,sim.s,sim.theta,sim.tx,sim.ty,s0,10);
    if(ref.success){res.s=ref.s;res.theta=ref.theta;res.tx=ref.tx;res.ty=ref.ty;res.n_inliers=ref.n_inliers;res.rms=ref.rms;res.inlier_mask=std::move(ref.inlier_mask);}
    else{res.s=sim.s;res.theta=sim.theta;res.tx=sim.tx;res.ty=sim.ty;res.n_inliers=inl.n_inliers;res.rms=inl.rms;res.inlier_mask=std::move(inl.inlier_mask);}
    res.success=true;
    res.n_phaseb_pairs=(int)filt.size(); res.n_phaseb_corr=(int)cu.size();
    fprintf(stderr,"[vm35] PhaseB OK: s=%.4f θ=%.2f° n=%d rms=%.3f corr=%zu\n",
            res.s,res.theta*RADTODEG,res.n_inliers,res.rms,cu.size());
    return res;
}

// ============================================================================
// Phase C: 全局NN匹配扩充（V3.5: 去掉Source 1 records去重，只保留Source 2）
// ============================================================================
struct Pair { int u, w; };

std::vector<Pair> expand_global_nn(
    const double* U, int N, const double* Wf, int M, double s0,
    double s_sol, double th_sol, double tx_sol, double ty_sol,
    const double* norm_U, const double* norm_Wf)
{
    std::vector<double> Wt(M*2);
    apply_similarity(Wf, M, s_sol, th_sol, tx_sol, ty_sol, Wt.data());

    std::unordered_set<uint64_t> seen;
    std::vector<Pair> candidates;

    // 全局NN匹配: 所有U→Wt, 角度+模过滤
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

    fprintf(stderr, "[vm35] PhaseC: %zu NN pairs\n", candidates.size());

    // 1-to-1 greedy: 按距离排序
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

    fprintf(stderr, "[vm35] PhaseC: %zu expanded (1to1)\n", expanded.size());
    return expanded;
}

// ============================================================================
// Phase D: 迭代中位数+3σ剔除（V3.5: 收敛保护）
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

    // 初始Umeyama
    {
        std::vector<double> sp, dp;
        for (auto& p : expanded) { sp.push_back(Wf[p.w*2]); sp.push_back(Wf[p.w*2+1]); dp.push_back(U[p.u*2]); dp.push_back(U[p.u*2+1]); }
        sim = umeyama(sp.data(), dp.data(), n);
        if (!sim.valid) return cr;
    }

    int iter = 0;
    int total_removed = 0;
    int initial_n = n;
    do {
        int n_removed = 0;

        // 计算当前保留数
        int n_keep = 0;
        for (int i = 0; i < n; ++i) if (keep[i]) n_keep++;

        // V3.5: 保留数 < 10时停止迭代
        if (n_keep < 10) {
            fprintf(stderr, "[vm35] PhaseD: 保留数 %d < 10, 停止迭代\n", n_keep);
            break;
        }

        // 应用变换
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

        // 中位数 + MAD
        auto dxc = dx_list, dyc = dy_list;
        double mdx = vec_median(dxc), mdy = vec_median(dyc);
        for (auto& v : dxc) v = std::abs(v - mdx);
        for (auto& v : dyc) v = std::abs(v - mdy);
        double sig_x = 1.4826 * vec_median(dxc);
        double sig_y = 1.4826 * vec_median(dyc);
        if (sig_x < 1e-10) sig_x = 1e-10;
        if (sig_y < 1e-10) sig_y = 1e-10;

        // 3σ过滤
        for (int j = 0; j < nk; ++j) {
            if (std::abs(dx_list[j] - mdx) > 3.0 * sig_x || std::abs(dy_list[j] - mdy) > 3.0 * sig_y) {
                keep[kept_indices[j]] = false;
                n_removed++;
            }
        }
        total_removed += n_removed;

        // V3.5: 累计剔除率 > 30%时打印警告
        double removal_rate = (double)total_removed / initial_n;
        if (removal_rate > 0.3) {
            fprintf(stderr, "[vm35] PhaseD 警告: 累计剔除率 %.1f%% > 30%% (已剔除 %d/%d)\n",
                    removal_rate * 100.0, total_removed, initial_n);
        }

        if (n_removed > 0) {
            // 用剩余点重新拟合
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

    // 收集干净对
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

    // 用干净对做最终Umeyama
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

    fprintf(stderr, "[vm35] PhaseD: %d removed in %d iters, %zu clean pairs, MAD-RMS=%.3f\"\n",
            cr.n_removed, iter, cr.clean_u.size(), cr.mad_rms_arcsec);
    return cr;
}

// ============================================================================
// Phase D': 星点扩增（双向NN + 方向一致性 + 径向幅度一致性过滤）
// ============================================================================
struct ExpandResult {
    std::vector<int> expand_u, expand_w; // 扩增匹配对
    int n_mutual;     // 双向NN匹配对数
    int n_after_filter; // 径向过滤后对数
};

// 径向一致性过滤（方向一致性 + 径向幅度一致性）
// 输入: 匹配对的残差向量和径向向量
// 输出: 过滤后保留的索引
std::vector<int> radial_consistency_filter(
    const std::vector<double>& cos_theta_list,  // cos_θ值
    const std::vector<double>& r_list,           // 径向距离
    const std::vector<double>& v_mag_list,       // 残差幅度
    int n_bins, int fit_order, int n_iters)
{
    int n = (int)cos_theta_list.size();
    if (n < 10) {
        // 点太少，全部保留
        std::vector<int> idx(n);
        std::iota(idx.begin(), idx.end(), 0);
        return idx;
    }

    std::vector<bool> keep(n, true);

    // Step 2: 方向一致性过滤
    {
        auto cos_vals = cos_theta_list;
        double med_cos = vec_median(cos_vals);
        std::vector<double> dev(n);
        for (int i = 0; i < n; ++i) dev[i] = std::abs(cos_vals[i] - med_cos);
        double mad_cos = vec_median(dev);
        double sig_cos = 1.4826 * mad_cos;
        if (sig_cos < 1e-10) sig_cos = 1e-10;
        for (int i = 0; i < n; ++i) {
            if (std::abs(cos_theta_list[i] - med_cos) > 3.0 * sig_cos) {
                keep[i] = false;
            }
        }
        int n_dir = 0;
        for (int i = 0; i < n; ++i) if (keep[i]) n_dir++;
        fprintf(stderr, "[vm35] PhaseD' 方向过滤: %d → %d\n", n, n_dir);
    }

    // Step 3: 径向幅度一致性过滤（迭代）
    for (int it = 0; it < n_iters; ++it) {
        // 收集保留点的(r, |v|)
        std::vector<std::pair<double,double>> rv;
        std::vector<int> rv_idx;
        for (int i = 0; i < n; ++i) {
            if (!keep[i]) continue;
            rv.push_back({r_list[i], v_mag_list[i]});
            rv_idx.push_back(i);
        }
        int nr = (int)rv.size();
        if (nr < 10) break;

        // 按r排序分bin
        std::vector<int> sort_idx(nr);
        std::iota(sort_idx.begin(), sort_idx.end(), 0);
        std::sort(sort_idx.begin(), sort_idx.end(), [&rv](int a, int b){ return rv[a].first < rv[b].first; });

        // 每bin内median+MAD3σ过滤
        int bins = std::min(n_bins, nr / 3);
        if (bins < 2) bins = 2;
        int bin_size = nr / bins;

        for (int b = 0; b < bins; ++b) {
            int start = b * bin_size;
            int end = (b == bins - 1) ? nr : (b + 1) * bin_size;
            if (end - start < 3) continue;

            // 收集bin内|v|值
            std::vector<double> v_vals;
            for (int k = start; k < end; ++k) v_vals.push_back(rv[sort_idx[k]].second);
            double med_v = vec_median(v_vals);
            std::vector<double> dev_v(v_vals.size());
            for (size_t k = 0; k < v_vals.size(); ++k) dev_v[k] = std::abs(v_vals[k] - med_v);
            double mad_v = vec_median(dev_v);
            double sig_v = 1.4826 * mad_v;
            if (sig_v < 1e-10) sig_v = 1e-10;

            // 过滤bin内离群点
            for (int k = start; k < end; ++k) {
                if (std::abs(rv[sort_idx[k]].second - med_v) > 3.0 * sig_v) {
                    keep[rv_idx[sort_idx[k]]] = false;
                }
            }
        }

        // 拟合|v|=f(r)的fit_order阶多项式
        std::vector<double> r_keep, v_keep;
        std::vector<int> keep_idx;
        for (int i = 0; i < n; ++i) {
            if (!keep[i]) continue;
            r_keep.push_back(r_list[i]);
            v_keep.push_back(v_mag_list[i]);
            keep_idx.push_back(i);
        }
        int nk = (int)r_keep.size();
        if (nk < fit_order + 1) break;

        // 构建Vandermonde矩阵
        Eigen::MatrixXd A(nk, fit_order + 1);
        Eigen::VectorXd b(nk);
        for (int i = 0; i < nk; ++i) {
            double rp = 1.0;
            for (int j = 0; j <= fit_order; ++j) {
                A(i, j) = rp;
                rp *= r_keep[i];
            }
            b(i) = v_keep[i];
        }
        Eigen::VectorXd coeffs = (A.transpose() * A).ldlt().solve(A.transpose() * b);

        // 残差MAD3σ过滤
        std::vector<double> residuals(nk);
        for (int i = 0; i < nk; ++i) {
            double pred = 0;
            double rp = 1.0;
            for (int j = 0; j <= fit_order; ++j) {
                pred += coeffs(j) * rp;
                rp *= r_keep[i];
            }
            residuals[i] = v_keep[i] - pred;
        }
        auto res_copy = residuals;
        double med_res = vec_median(res_copy);
        std::vector<double> dev_res(nk);
        for (int i = 0; i < nk; ++i) dev_res[i] = std::abs(residuals[i] - med_res);
        double mad_res = vec_median(dev_res);
        double sig_res = 1.4826 * mad_res;
        if (sig_res < 1e-10) sig_res = 1e-10;

        for (int i = 0; i < nk; ++i) {
            if (std::abs(residuals[i] - med_res) > 3.0 * sig_res) {
                keep[keep_idx[i]] = false;
            }
        }

        int n_after = 0;
        for (int i = 0; i < n; ++i) if (keep[i]) n_after++;
        fprintf(stderr, "[vm35] PhaseD' 径向幅度过滤 iter %d: %d 保留\n", it, n_after);
    }

    // 收集保留索引
    std::vector<int> result;
    for (int i = 0; i < n; ++i) if (keep[i]) result.push_back(i);
    return result;
}

ExpandResult expand_star_pairs(
    const double* U, int N_img, const double* Wf, int M,
    double s_sol, double theta_sol, double tx_sol, double ty_sol,
    double s0, double img_w, double img_h,
    int expand_n_gaia, int expand_n_img,
    int radial_n_bins, int radial_fit_order, int radial_n_iters)
{
    ExpandResult er;
    er.n_mutual = 0;
    er.n_after_filter = 0;

    // 转换到像素空间做匹配（更直观的距离度量）
    double cx = img_w / 2.0, cy = img_h / 2.0;

    // 投影Gaia星像素坐标
    std::vector<double> Wt(M*2);
    apply_similarity(Wf, M, s_sol, theta_sol, tx_sol, ty_sol, Wt.data());
    std::vector<double> gaia_px(M*2);
    for (int i = 0; i < M; ++i) {
        gaia_px[i*2]   = Wt[i*2] / s0 + cx;
        gaia_px[i*2+1] = -Wt[i*2+1] / s0 + cy;  // Y翻转
    }

    // 图像星像素坐标
    std::vector<double> img_px(N_img*2);
    for (int i = 0; i < N_img; ++i) {
        img_px[i*2]   = U[i*2] / s0 + cx;
        img_px[i*2+1] = -U[i*2+1] / s0 + cy;  // Y翻转
    }

    // Step 1: 双向NN匹配
    int n_gaia_use = std::min(expand_n_gaia, M);
    int n_img_use = std::min(expand_n_img, N_img);

    // 选取离图像中心最近的Gaia投影星（像素空间）
    std::vector<int> gaia_idx(n_gaia_use);
    {
        std::vector<std::pair<double,int>> dist_idx(M);
        for (int i = 0; i < M; ++i) {
            double dx = gaia_px[i*2] - cx, dy = gaia_px[i*2+1] - cy;
            dist_idx[i] = {dx*dx+dy*dy, i};
        }
        std::partial_sort(dist_idx.begin(), dist_idx.begin()+n_gaia_use, dist_idx.end());
        for (int i = 0; i < n_gaia_use; ++i) gaia_idx[i] = dist_idx[i].second;
    }

    // 选取离图像中心最近的图像星（像素空间）
    std::vector<int> img_idx(n_img_use);
    {
        std::vector<std::pair<double,int>> dist_idx(N_img);
        for (int i = 0; i < N_img; ++i) {
            double dx = img_px[i*2] - cx, dy = img_px[i*2+1] - cy;
            dist_idx[i] = {dx*dx+dy*dy, i};
        }
        std::partial_sort(dist_idx.begin(), dist_idx.begin()+n_img_use, dist_idx.end());
        for (int i = 0; i < n_img_use; ++i) img_idx[i] = dist_idx[i].second;
    }

    // 正向KDTree: 每颗投影Gaia星→最近邻图像星 (像素空间)
    PointCloud2D img_cloud; img_cloud.pts.resize(n_img_use);
    for (int i = 0; i < n_img_use; ++i) img_cloud.pts[i] = {img_px[img_idx[i]*2], img_px[img_idx[i]*2+1]};
    KDTree img_tree(2, img_cloud, nanoflann::KDTreeSingleIndexAdaptorParams(10));

    // 反向KDTree: 每颗图像星→最近邻投影Gaia星 (像素空间)
    PointCloud2D gaia_cloud; gaia_cloud.pts.resize(n_gaia_use);
    for (int i = 0; i < n_gaia_use; ++i) gaia_cloud.pts[i] = {gaia_px[gaia_idx[i]*2], gaia_px[gaia_idx[i]*2+1]};
    KDTree gaia_tree(2, gaia_cloud, nanoflann::KDTreeSingleIndexAdaptorParams(10));

    // 正向: Gaia→图像 (像素空间)
    std::vector<int> fwd_gaia_to_img(n_gaia_use);
    std::vector<double> fwd_dist(n_gaia_use);
    for (int i = 0; i < n_gaia_use; ++i) {
        double q[2] = {gaia_px[gaia_idx[i]*2], gaia_px[gaia_idx[i]*2+1]};
        KDTreeIndexType idx; double ds;
        nanoflann::KNNResultSet<double,KDTreeIndexType> rs(1); rs.init(&idx,&ds);
        img_tree.findNeighbors(rs, q);
        fwd_gaia_to_img[i] = (int)idx;
        fwd_dist[i] = std::sqrt(ds);
    }

    // 反向: 图像→Gaia (像素空间)
    std::vector<int> bwd_img_to_gaia(n_img_use);
    std::vector<double> bwd_dist(n_img_use);
    for (int i = 0; i < n_img_use; ++i) {
        double q[2] = {img_px[img_idx[i]*2], img_px[img_idx[i]*2+1]};
        KDTreeIndexType idx; double ds;
        nanoflann::KNNResultSet<double,KDTreeIndexType> rs(1); rs.init(&idx,&ds);
        gaia_tree.findNeighbors(rs, q);
        bwd_img_to_gaia[i] = (int)idx;
        bwd_dist[i] = std::sqrt(ds);
    }

    // 互为最近邻的对
    struct MutualPair { int gaia_local, img_local; double dist; };
    std::vector<MutualPair> mutual_pairs;
    // 调试: 统计NN距离分布
    double fwd_dist_med = 0;
    {
        std::vector<double> fd(fwd_dist.begin(), fwd_dist.end());
        std::sort(fd.begin(), fd.end());
        fwd_dist_med = fd[fd.size()/2];
    }
    fprintf(stderr, "[vm35] PhaseD' 正向NN距离: med=%.2f min=%.2f max=%.2f\n",
            fwd_dist_med, *std::min_element(fwd_dist.begin(), fwd_dist.end()),
            *std::max_element(fwd_dist.begin(), fwd_dist.end()));
    for (int i = 0; i < n_gaia_use; ++i) {
        int img_l = fwd_gaia_to_img[i];
        // 检查反向是否也指向自己
        if (img_l >= 0 && img_l < n_img_use && bwd_img_to_gaia[img_l] == i) {
            mutual_pairs.push_back({i, img_l, fwd_dist[i]});
        }
    }
    er.n_mutual = (int)mutual_pairs.size();
    fprintf(stderr, "[vm35] PhaseD' 双向NN: %d 互为最近邻对 (n_gaia=%d n_img=%d)\n",
            er.n_mutual, n_gaia_use, n_img_use);

    if (mutual_pairs.size() < 5) {
        fprintf(stderr, "[vm35] PhaseD' 互为最近邻对太少，跳过扩增\n");
        return er;
    }

    // 计算每对的残差向量和径向向量
    std::vector<double> cos_theta_list, r_list, v_mag_list;
    std::vector<int> mutual_gaia_global, mutual_img_global;
    for (auto& mp : mutual_pairs) {
        int g_global = gaia_idx[mp.gaia_local];
        int i_global = img_idx[mp.img_local];

        // predicted_pixel = gaia_px[g_global]
        double pred_x = gaia_px[g_global*2], pred_y = gaia_px[g_global*2+1];
        // det_pixel = img_px[i_global]
        double det_x = img_px[i_global*2], det_y = img_px[i_global*2+1];

        // 径向向量: r_vec = predicted_pixel - center
        double rx = pred_x - cx, ry = pred_y - cy;
        double r_mag = std::sqrt(rx*rx + ry*ry);
        if (r_mag < 1e-10) continue; // 中心点跳过

        // 残差向量: v = det_pixel - predicted_pixel
        double vx = det_x - pred_x, vy = det_y - pred_y;
        double v_mag = std::sqrt(vx*vx + vy*vy);

        // cos_θ = dot(r_hat, v_hat)
        double cos_th = (rx * vx + ry * vy) / (r_mag * std::max(v_mag, 1e-10));

        cos_theta_list.push_back(cos_th);
        r_list.push_back(r_mag);
        v_mag_list.push_back(v_mag);
        mutual_gaia_global.push_back(g_global);
        mutual_img_global.push_back(i_global);
    }

    // Step 2+3: 径向一致性过滤
    auto kept = radial_consistency_filter(cos_theta_list, r_list, v_mag_list,
                                           radial_n_bins, radial_fit_order, radial_n_iters);

    er.n_after_filter = (int)kept.size();
    fprintf(stderr, "[vm35] PhaseD' 径向过滤后: %d 对\n", er.n_after_filter);

    // 1-to-1去重
    std::vector<std::tuple<double,int,int>> scored;
    for (int ki : kept) {
        int g = mutual_gaia_global[ki], u = mutual_img_global[ki];
        double dx = img_px[u*2] - gaia_px[g*2], dy = img_px[u*2+1] - gaia_px[g*2+1];
        scored.push_back({dx*dx+dy*dy, u, g});
    }
    std::sort(scored.begin(), scored.end());

    std::vector<int> u_used(N_img, 0), w_used(M, 0);
    for (auto& [dsq, u, w] : scored) {
        if (u_used[u] || w_used[w]) continue;
        u_used[u] = 1; w_used[w] = 1;
        er.expand_u.push_back(u);
        er.expand_w.push_back(w);
    }

    fprintf(stderr, "[vm35] PhaseD' 扩增完成: %zu 对 (1to1去重后)\n", er.expand_u.size());
    return er;
}

// ============================================================================
// Phase E: 自适应阶数SIP多项式拟合
// ============================================================================
struct SIPResult {
    double sip_A[36];
    double sip_B[36];
    double cd[4];
    double crval[2];
    double crpix[2];
    double rms_px;
    int order; // 实际使用的SIP阶数
    // 迭代优化后的变换参数（SIP+s联合收敛）
    double s_refined;
    double theta_refined;
    double tx_refined;
    double ty_refined;
};

// 返回(p,q)项在多项式基中的索引（按阶数排序）
int poly_index(int p, int q, int max_order) {
    int idx = 0;
    for (int o = 0; o <= max_order; ++o)
        for (int pp = 0; pp <= o; ++pp) {
            int qq = o - pp;
            if (pp == p && qq == q) return idx;
            idx++;
        }
    return -1;
}

// 给定最大阶数，返回多项式项数
int poly_nterms(int max_order) {
    return (max_order + 1) * (max_order + 2) / 2;
}

SIPResult fit_affine_sip_adaptive(
    const double* U, const double* Wf,
    const std::vector<int>& clean_u, const std::vector<int>& clean_w,
    double s0, double w, double h,
    double center_ra, double center_dec,
    double s_sol, double theta_sol,
    int flip_mode,
    int sip_order,  // V3.5: 自适应SIP阶数
    const char* wcs_out_path)
{
    SIPResult sr;
    std::memset(sr.sip_A, 0, sizeof(sr.sip_A));
    std::memset(sr.sip_B, 0, sizeof(sr.sip_B));
    std::memset(sr.cd, 0, sizeof(sr.cd));
    sr.crval[0] = center_ra; sr.crval[1] = center_dec;
    sr.crpix[0] = w / 2.0; sr.crpix[1] = h / 2.0;
    sr.rms_px = 0;
    sr.order = sip_order;
    sr.s_refined = s_sol;
    sr.theta_refined = theta_sol;
    sr.tx_refined = 0;
    sr.ty_refined = 0;

    int M_D = (int)clean_u.size();

    bool fx = (flip_mode == 1 || flip_mode == 3);
    bool fy = (flip_mode == 2 || flip_mode == 3);

    // ============================================================
    // Pre-filter: 向量残差中值离群排异
    //   对全部匹配对做Umeyama→残差→MAD3σ过滤，剔除少量假匹配
    // ============================================================
    std::vector<int> clean_u_f = clean_u;
    std::vector<int> clean_w_f = clean_w;
    {
        std::vector<double> wf_src(M_D*2), u_dst(M_D*2);
        for(int i=0;i<M_D;++i){
            wf_src[i*2]=Wf[clean_w_f[i]*2]; wf_src[i*2+1]=Wf[clean_w_f[i]*2+1];
            u_dst[i*2]=U[clean_u_f[i]*2]; u_dst[i*2+1]=U[clean_u_f[i]*2+1];
        }
        auto pre_sim = umeyama(wf_src.data(), u_dst.data(), M_D);
        if(pre_sim.valid){
            std::vector<double> res_dx(M_D), res_dy(M_D);
            double ct_pre=std::cos(pre_sim.theta), st_pre=std::sin(pre_sim.theta);
            for(int i=0;i<M_D;++i){
                int ui=clean_u_f[i], wi=clean_w_f[i];
                double wx0=Wf[wi*2], wy0=Wf[wi*2+1];
                double wtx=pre_sim.s*(ct_pre*wx0-st_pre*wy0)+pre_sim.tx;
                double wty=pre_sim.s*(st_pre*wx0+ct_pre*wy0)+pre_sim.ty;
                res_dx[i]=U[ui*2]-wtx;
                res_dy[i]=U[ui*2+1]-wty;
            }
            auto mad_pair = [](std::vector<double>& v) -> std::pair<double,double> {
                size_t n=v.size(); if(n<3)return std::make_pair(0.0,1e30);
                std::vector<double> c=v; std::nth_element(c.begin(),c.begin()+n/2,c.end());
                double med=c[n/2];
                std::vector<double> ads(n);
                for(size_t i=0;i<n;++i)ads[i]=std::abs(v[i]-med);
                std::nth_element(ads.begin(),ads.begin()+n/2,ads.end());
                return std::make_pair(med,1.4826*ads[n/2]);
            };
            auto [mdx,sdx] = mad_pair(res_dx);
            auto [mdy,sdy] = mad_pair(res_dy);
            int n_pre=(int)M_D;
            clean_u_f.clear(); clean_w_f.clear();
            for(int i=0;i<n_pre;++i){
                double d = std::max(std::abs(res_dx[i]-mdx)/std::max(sdx,1e-10),
                                     std::abs(res_dy[i]-mdy)/std::max(sdy,1e-10));
                if(d < 3.0){
                    clean_u_f.push_back(clean_u[i]);
                    clean_w_f.push_back(clean_w[i]);
                }
            }
            fprintf(stderr,"[vm35] Pre-filter(向量中值离群): M_D=%d → %zu 对 (med_dx=%.3f\" σ_dx=%.3f\" med_dy=%.3f\" σ_dy=%.3f\")\n",
                    n_pre,clean_u_f.size(),mdx,sdx,mdy,sdy);
            M_D = (int)clean_u_f.size();
        } else {
            fprintf(stderr,"[vm35] Pre-filter: umeyama invalid (M_D=%d), skip\n", M_D);
        }
    }

    // V3.5: 始终执行Layer 0+1(CD+仿射)，仅order=0时跳过Layer 2(SIP)
    if (M_D < 2) {
        // 不足2对，回退纯Phase B CD
        double ct = std::cos(theta_sol), st = std::sin(theta_sol);
        double cos_dec = std::cos(center_dec * DEGTORAD);
        if (cos_dec < 1e-10) cos_dec = 1e-10;
        double s0_over_s_3600 = s0 / (s_sol * 3600.0);
        double sign_x = fx ? -1.0 : 1.0;
        double sign_y = fy ? -1.0 : 1.0;
        sr.cd[0] = sign_x * s0_over_s_3600 * ct / cos_dec;
        sr.cd[1] = -sign_x * s0_over_s_3600 * st / cos_dec;
        sr.cd[2] = -sign_y * s0_over_s_3600 * st;
        sr.cd[3] = -sign_y * s0_over_s_3600 * ct;
        fprintf(stderr, "[vm35] PhaseE: M_D=%d<2, fallback PhaseB CD\n", M_D);
        if(wcs_out_path&&wcs_out_path[0]){
            FILE*fw=fopen(wcs_out_path,"w");
            if(fw){
                fprintf(fw,"{\n  \"CD\": [[%.16e, %.16e], [%.16e, %.16e]],\n",
                        sr.cd[0],sr.cd[1],sr.cd[2],sr.cd[3]);
                fprintf(fw,"  \"CRVAL\": [%.10f, %.10f],\n  \"CRPIX\": [%.3f, %.3f],\n",
                        center_ra,center_dec,w/2.0,h/2.0);
                fprintf(fw,"  \"SIP_A\": [");for(int i=0;i<36;++i)fprintf(fw,"%s0.0",i?",":"");fprintf(fw,"],\n  \"SIP_B\": [");
                for(int i=0;i<36;++i)fprintf(fw,"%s0.0",i?",":"");fprintf(fw,"],\n  \"RMS_PX\": 0.000000,\n  \"SIP_ORDER\": 0,\n");
                fprintf(fw,"  \"MATCH_PAIRS\": [");
                double cos_cdc = std::cos(center_dec * DEGTORAD);
                for(int pi=0;pi<M_D;++pi){
                    int ui=clean_u_f[pi],wi=clean_w_f[pi];
                    double ipx=U[ui*2]/s0 + w/2.0;
                    double ipy=-U[ui*2+1]/s0 + h/2.0;
                    double wfx=fx?-Wf[wi*2]:Wf[wi*2];
                    double wfy=fy?-Wf[wi*2+1]:Wf[wi*2+1];
                    double gaia_ra=center_ra+wfx/(3600.0*cos_cdc);
                    double gaia_dec=center_dec+wfy/3600.0;
                    fprintf(fw,"%s[%.3f,%.3f,%.10f,%.10f]",pi?",":"",ipx,ipy,gaia_ra,gaia_dec);
                }
                fprintf(fw,"]\n}\n");
                fclose(fw);
                fprintf(stderr,"[vm35] WCS JSON written to: %s\n",wcs_out_path);
            }
        }
        return sr;
    }

    double crpix_x = w / 2.0, crpix_y = h / 2.0;
    double sign_x = fx ? -1.0 : 1.0;
    double sign_y = fy ? -1.0 : 1.0;

    // ============================================================
    // MAD稳健分层拟合:
    //   Layer 0: Umeyama弧秒 → CD(符号验正→选RMS更低)
    //   Layer 1: 像素残差MAD剔除outlier(3轮) → 仿射/CD/CRVAL
    //   Layer 2: SIP BIC选阶(仅≥2阶)
    // ============================================================
    double cos_dec0 = std::cos(center_dec * DEGTORAD);
    if (cos_dec0 < 1e-10) cos_dec0 = 1e-10;

    // ---- 构建弧秒src/dst ----
    std::vector<double> us0(M_D*2), ws0(M_D*2);
    for (int i=0;i<M_D;++i){
        us0[i*2]=U[clean_u_f[i]*2]; us0[i*2+1]=U[clean_u_f[i]*2+1];
        ws0[i*2]=Wf[clean_w_f[i]*2]; ws0[i*2+1]=Wf[clean_w_f[i]*2+1];
    }
    auto au0=umeyama(ws0.data(),us0.data(),M_D);
    if(!au0.valid){au0.s=s_sol; au0.theta=theta_sol;}

    // ---- 符号验正: Umeyama有 ±π 歧义, 选RMS更低的 ----
    double rms_best=1e30; int best_sign=0;
    double CD_best[4], s_best=0, th_best=0, s3600=s0/(au0.s*3600.0);
    for(int sg=0;sg<2;++sg){
        double th_try=sg?au0.theta+PI:au0.theta;
        double ct=std::cos(th_try),st=std::sin(th_try);
        double cd_t[4]={  sign_x*s3600*ct/cos_dec0, -sign_x*s3600*st/cos_dec0,
                         -sign_y*s3600*st,          -sign_y*s3600*ct };
        double cd=cd_t[0]*cd_t[3]-cd_t[1]*cd_t[2];
        double cdi[4]={cd_t[3]/cd,-cd_t[1]/cd,-cd_t[2]/cd,cd_t[0]/cd};
        double rms=0;
        for(int i=0;i<M_D;++i){
            int wi=clean_w_f[i];
            double Wx=fx?-Wf[wi*2]:Wf[wi*2],Wy=fy?-Wf[wi*2+1]:Wf[wi*2+1];
            double xi=cdi[0]*Wx/(3600.*cos_dec0)+cdi[1]*Wy/3600.;
            double eta=cdi[2]*Wx/(3600.*cos_dec0)+cdi[3]*Wy/3600.;
            double sx=U[clean_u_f[i]*2]/s0+crpix_x-sr.crpix[0];
            double sy=-U[clean_u_f[i]*2+1]/s0+crpix_y-sr.crpix[1];
            double dx=xi-sx,dy=eta-sy; rms+=dx*dx+dy*dy;
        }
        rms=std::sqrt(rms/M_D);
        if(rms<rms_best){rms_best=rms;best_sign=sg;s_best=au0.s;th_best=th_try;
            for(int k=0;k<4;++k)CD_best[k]=cd_t[k];}
    }
    for(int k=0;k<4;++k)sr.cd[k]=CD_best[k];
    double cd21=sr.cd[0]*sr.cd[3]-sr.cd[1]*sr.cd[2];
    double cdi2[4]={sr.cd[3]/cd21,-sr.cd[1]/cd21,-sr.cd[2]/cd21,sr.cd[0]/cd21};

    fprintf(stderr,"[vm35] Umeyama: s=%.6f θ=%.4f° sign=%d CD_RMS=%.4fpx\n",
            s_best,th_best*RADTODEG,best_sign,rms_best);

    // ---- 像素src/dst并MAD迭代剔除outlier ----
    std::vector<double> ssx(M_D),ssy(M_D),ddx(M_D),ddy(M_D);
    std::vector<bool> keep(M_D,true);
    for(int i=0;i<M_D;++i){
        int ui=clean_u_f[i],wi=clean_w_f[i];
        ssx[i]=U[ui*2]/s0+crpix_x-sr.crpix[0];
        ssy[i]=-U[ui*2+1]/s0+crpix_y-sr.crpix[1];
        double Wx=fx?-Wf[wi*2]:Wf[wi*2],Wy=fy?-Wf[wi*2+1]:Wf[wi*2+1];
        ddx[i]=cdi2[0]*Wx/(3600.*cos_dec0)+cdi2[1]*Wy/3600.;
        ddy[i]=cdi2[2]*Wx/(3600.*cos_dec0)+cdi2[3]*Wy/3600.;
    }

    for(int mad_iter=0;mad_iter<3;++mad_iter){
        // 当前keep点残差中位数绝对值
        std::vector<double> rlist;
        for(int i=0;i<M_D;++i)if(keep[i]){
            double rx=ddx[i]-ssx[i],ry=ddy[i]-ssy[i];
            rlist.push_back(std::sqrt(rx*rx+ry*ry));
        }
        if(rlist.size()<10)break;
        std::sort(rlist.begin(),rlist.end());
        double mad=rlist[rlist.size()/2]*1.4826; // MAD→sigma
        double thresh=std::max(5.0,3.0*mad); // 至少5px门槛
        int removed=0;
        for(int i=0;i<M_D;++i)if(keep[i]){
            double rx=ddx[i]-ssx[i],ry=ddy[i]-ssy[i];
            if(std::sqrt(rx*rx+ry*ry)>thresh){keep[i]=false;removed++;}
        }
        fprintf(stderr,"[vm35] MAD轮%d: keep=%zu thresh=%.1fpx removed=%d\n",
                mad_iter,std::count(keep.begin(),keep.end(),true),thresh,removed);
        if(removed==0)break;
    }

    int M_clean=0;
    for(int i=0;i<M_D;++i)if(keep[i])M_clean++;
    if(M_clean<5){fprintf(stderr,"[vm35] MAD后点数不足\n");return sr;}

    // ---- 全仿射(仅clean点, fit: det = A·gaia + t) ----
    Eigen::MatrixXd Lc(M_clean*2,6); Eigen::VectorXd Rc(M_clean*2);
    int row=0;
    for(int i=0;i<M_D;++i)if(keep[i]){
        double xx=ssx[i],yy=ssy[i];
        double gx=ddx[i],gy=ddy[i];
        Lc(row*2,0)=gx;Lc(row*2,1)=gy;Lc(row*2,2)=1;Lc(row*2,3)=0;Lc(row*2,4)=0;Lc(row*2,5)=0;
        Lc(row*2+1,0)=0;Lc(row*2+1,1)=0;Lc(row*2+1,2)=0;Lc(row*2+1,3)=gx;Lc(row*2+1,4)=gy;Lc(row*2+1,5)=1;
        Rc(row*2)=xx;Rc(row*2+1)=yy;
        row++;
    }
    Eigen::VectorXd ab = (Lc.transpose()*Lc).ldlt().solve(Lc.transpose()*Rc);
    double a00=ab(0),a01=ab(1),tx=ab(2),a10=ab(3),a11=ab(4),ty=ab(5);

    // 仿射: det = A·gaia + t
    // CD'·det = CD·gaia → CD'·(A·gaia+t) = CD·gaia
    // → CD'·A = CD → CD' = CD·A⁻¹
    // → CRVAL' = center − CD'·t
    double idet=a00*a11-a01*a10;
    double ai00=a11/idet, ai01=-a01/idet, ai10=-a10/idet, ai11=a00/idet;
    double c00=sr.cd[0]*ai00+sr.cd[1]*ai10, c01=sr.cd[0]*ai01+sr.cd[1]*ai11;
    double c10=sr.cd[2]*ai00+sr.cd[3]*ai10, c11=sr.cd[2]*ai01+sr.cd[3]*ai11;
    sr.cd[0]=c00;sr.cd[1]=c01;sr.cd[2]=c10;sr.cd[3]=c11;
    sr.crval[0]=center_ra-(c00*tx+c01*ty);
    sr.crval[1]=center_dec-(c10*tx+c11*ty);
    fprintf(stderr,"[vm35] 仿射CRVAL更新: center_ra=%.7f crval=%.7f Δ=%.6fdeg tx=%.1fpx\n",
            center_ra, sr.crval[0], center_ra-sr.crval[0], tx);

    // 新CD重投影 → 最终仿射残差
    double cd2=sr.cd[0]*sr.cd[3]-sr.cd[1]*sr.cd[2];
    double cd_i[4]={sr.cd[3]/cd2,-sr.cd[1]/cd2,-sr.cd[2]/cd2,sr.cd[0]/cd2};
    double rms_aff=0;
    for(int i=0;i<M_D;++i)if(keep[i]){
        int wi=clean_w_f[i];
        double Wx=fx?-Wf[wi*2]:Wf[wi*2],Wy=fy?-Wf[wi*2+1]:Wf[wi*2+1];
        double xi=cd_i[0]*Wx/(3600.*cos_dec0)+cd_i[1]*Wy/3600.;
        double eta=cd_i[2]*Wx/(3600.*cos_dec0)+cd_i[3]*Wy/3600.;
        ddx[i]=xi;ddy[i]=eta;
        double rx=ddx[i]-ssx[i],ry=ddy[i]-ssy[i];
        rms_aff+=rx*rx+ry*ry;
    }
    rms_aff=std::sqrt(rms_aff/M_clean);

    // 残差(仅clean点用于SIP)
    std::vector<int> clean_idx; clean_idx.reserve(M_clean);
    for(int i=0;i<M_D;++i)if(keep[i]){
        ddx[i]-=ssx[i];ddy[i]-=ssy[i];
        clean_idx.push_back(i);
    }

    double cos_cr=std::cos(sr.crval[1]*DEGTORAD);if(cos_cr<1e-10)cos_cr=1e-10;
    sr.s_refined=s0/(3600.*std::sqrt(std::abs(cd2)*cos_cr));
    sr.theta_refined=std::atan2(sign_y?-sr.cd[2]:sr.cd[2],sign_x?-sr.cd[0]:sr.cd[0]);
    sr.tx_refined=(center_ra-sr.crval[0])*cos_cr*3600.;
    sr.ty_refined=(center_dec-sr.crval[1])*3600.;

    fprintf(stderr,"[vm35] 仿射(MAD后%d点): A=[%.6f,%.6f;%.6f,%.6f] t=[%.3f,%.3f]px RMS=%.4fpx\n",
            M_clean,a00,a01,a10,a11,tx,ty,rms_aff);

    // ---- SIP BIC（仅sip_order>0时执行） ----
    if (sip_order > 0 && M_clean >= 5) {
    double xs=w/2.0,ys=h/2.0;
    std::vector<double> nu(clean_idx.size()),nv(clean_idx.size());
    std::vector<double> rx_h(clean_idx.size()),ry_h(clean_idx.size());
    for(size_t k=0;k<clean_idx.size();++k){
        int i=clean_idx[k];
        nu[k]=ssx[i]/xs;nv[k]=ssy[i]/ys;
        rx_h[k]=ddx[i];ry_h[k]=ddy[i];
    }
    int Mc=(int)clean_idx.size();

    int max_order=sip_order, best_order=0;
    double best_bic=1e30, best_rms=0;
    double best_sA[36]={0},best_sB[36]={0};

    for(int try_o=2;try_o<=max_order;++try_o){
        int nhi=poly_nterms(try_o)-3; if(Mc<=nhi)continue;
        Eigen::MatrixXd Ah(Mc,nhi);Eigen::VectorXd bxh(Mc),byh(Mc);
        for(int j=0;j<Mc;++j){
            bxh(j)=rx_h[j];byh(j)=ry_h[j];int co=0;
            for(int o=2;o<=try_o;++o)for(int p=0;p<=o;++p)
                Ah(j,co++)=std::pow(nu[j],p)*std::pow(nv[j],o-p);
        }
        Eigen::VectorXd bxa=(Ah.transpose()*Ah).ldlt().solve(Ah.transpose()*bxh);
        Eigen::VectorXd bya=(Ah.transpose()*Ah).ldlt().solve(Ah.transpose()*byh);
        double ssq=(bxh-Ah*bxa).squaredNorm()+(byh-Ah*bya).squaredNorm();
        double rms_h=std::sqrt(ssq/Mc);
        int kp=nhi*2;double bic=Mc*std::log(ssq/Mc)+kp*std::log((double)Mc);
        fprintf(stderr,"[vm35] SIP阶=%d 项=%d RMS=%.4fpx BIC=%.2f\n",try_o,nhi,rms_h,bic);
        if(bic<best_bic){
            best_bic=bic;best_order=try_o;best_rms=rms_h;
            for(int o=2;o<=try_o;++o)for(int p=0;p<=o;++p){
                int q=o-p;if(p>=6||q>=6)continue;
                int hi=-1,cn=0;
                for(int oo=2;oo<=try_o;++oo)for(int pp=0;pp<=oo;++pp){
                    if(pp==p&&(oo-pp)==q){hi=cn;break;}cn++;
                }
                if(hi<0||hi>=nhi)continue;
                double nf=std::pow(xs,p)*std::pow(ys,q);
                best_sA[p*6+q]=bxa(hi)/nf;best_sB[p*6+q]=bya(hi)/nf;
            }
        }
    }
    sr.order=best_order;sr.rms_px=best_rms;
    for(int i=0;i<36;++i){sr.sip_A[i]=best_sA[i];sr.sip_B[i]=best_sB[i];}

    fprintf(stderr,"[vm35] PhaseE: Ume_RMS=%.3f affine_RMS=%.3f SIP阶=%d RMS=%.3fpx s=%.6f θ=%.3f°\n",
            rms_best,rms_aff,best_order,best_rms,sr.s_refined,sr.theta_refined*RADTODEG);
    fprintf(stderr,"[vm35] CD=[[%.6e,%.6e],[%.6e,%.6e]] CRVAL=[%.10f,%.10f]\n",
            sr.cd[0],sr.cd[1],sr.cd[2],sr.cd[3],sr.crval[0],sr.crval[1]);

    // SIP系数打印
    int ndb=std::min(10,Mc);
    fprintf(stderr,"[vm35] SIP coeffs(first %d non-zero,order=%d):\n",ndb,sr.order);
    int np=0;
    for(int p=0;p<6&&np<ndb;++p)for(int q=0;q<6&&np<ndb;++q){
        if(p+q<2||p+q>sr.order)continue;
        if(std::abs(sr.sip_A[p*6+q])<1e-30&&std::abs(sr.sip_B[p*6+q])<1e-30)continue;
        fprintf(stderr,"[vm35]   A[%d][%d]=%.6e  B[%d][%d]=%.6e\n",p,q,sr.sip_A[p*6+q],p,q,sr.sip_B[p*6+q]);
        np++;
    }
    } else {
        sr.order=0; sr.rms_px=rms_aff;
        fprintf(stderr,"[vm35] PhaseE: Ume_RMS=%.3f affine_RMS=%.3f 跳过SIP(order=0) s=%.6f θ=%.3f°\n",
                rms_best,rms_aff,sr.s_refined,sr.theta_refined*RADTODEG);
        fprintf(stderr,"[vm35] CD=[[%.6e,%.6e],[%.6e,%.6e]] CRVAL=[%.10f,%.10f]\n",
                sr.cd[0],sr.cd[1],sr.cd[2],sr.cd[3],sr.crval[0],sr.crval[1]);
    }

    // JSON WCS输出
    if(wcs_out_path&&wcs_out_path[0]){
        FILE*fw=fopen(wcs_out_path,"w");
        if(fw){
            fprintf(fw,"{\n  \"CD\": [[%.16e, %.16e], [%.16e, %.16e]],\n",
                    sr.cd[0],sr.cd[1],sr.cd[2],sr.cd[3]);
            fprintf(fw,"  \"CRVAL\": [%.10f, %.10f],\n  \"CRPIX\": [%.3f, %.3f],\n",
                    sr.crval[0],sr.crval[1],sr.crpix[0],sr.crpix[1]);
            fprintf(fw,"  \"SIP_A\": [");
            for(int i=0;i<36;++i)fprintf(fw,"%s%.16e",i?",":"",sr.sip_A[i]);
            fprintf(fw,"],\n  \"SIP_B\": [");
            for(int i=0;i<36;++i)fprintf(fw,"%s%.16e",i?",":"",sr.sip_B[i]);
            fprintf(fw,"],\n  \"RMS_PX\": %.6f,\n  \"SIP_ORDER\": %d,\n",sr.rms_px,sr.order);
            fprintf(fw,"  \"MATCH_PAIRS\": [");
            double cos_cd = std::cos(center_dec * DEGTORAD);
            for(int pi=0;pi<M_D;++pi){
                int ui=clean_u_f[pi],wi=clean_w_f[pi];
                double ipx=U[ui*2]/s0 + w/2.0;
                double ipy=-U[ui*2+1]/s0 + h/2.0;
                double wfx=fx?-Wf[wi*2]:Wf[wi*2];
                double wfy=fy?-Wf[wi*2+1]:Wf[wi*2+1];
                double gaia_ra = center_ra + wfx / (3600.0 * cos_cd);
                double gaia_dec = center_dec + wfy / 3600.0;
                fprintf(fw,"%s[%.3f,%.3f,%.10f,%.10f]",pi?",":"",ipx,ipy,gaia_ra,gaia_dec);
            }
            fprintf(fw,"]\n}\n");
            fclose(fw);
            fprintf(stderr,"[vm35] WCS JSON written to: %s\n",wcs_out_path);
        }else fprintf(stderr,"[vm35] ERROR: cannot open WCS output: %s\n",wcs_out_path);
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
    V35PhaseABResult ab;
};

ModeRes solve_single_mode(const double* U, int N, const double* W, int M,
                           int mode, double s0, double s_min, double s_max,
                           int K_total, int batch_size, int min_samples,
                           int min_inliers, int seed, double fov_diag_asec,
                           volatile std::atomic<bool>* early_exit)
{
    ModeRes mr; mr.success=false; mr.norm_score=0; mr.peak_snr=0; mr.n_samples=0; mr.mode=mode;
    if(early_exit && early_exit->load(std::memory_order_relaxed)){fprintf(stderr,"[vm35] mode%d: skip\n",mode);return mr;}
    std::vector<double> Wf(M*2); apply_flip(W,M,mode,Wf.data());
    fprintf(stderr,"[vm35] mode%d: N=%d M=%d\n",mode,N,M);
    mr.ab=record_and_filter(U,N,Wf.data(),M,s0,s_min,s_max,K_total,batch_size,min_samples,min_inliers,seed+mode,fov_diag_asec);
    mr.peak_snr=mr.ab.peak_snr; mr.n_samples=mr.ab.n_samples;
    if(!mr.ab.success){fprintf(stderr,"[vm35] mode%d: fail\n",mode);return mr;}
    mr.s=mr.ab.s; mr.theta=mr.ab.theta; mr.tx=mr.ab.tx; mr.ty=mr.ab.ty;
    mr.n_inliers=mr.ab.n_inliers; mr.rms=mr.ab.rms;
    mr.inlier_mask=std::move(mr.ab.inlier_mask);
    mr.norm_score=compute_normalized_score(mr.n_inliers,mr.rms,N,M,1.0*s0);
    mr.success=true;
    if(mr.success&&mr.s>=0.9&&mr.s<=1.1&&mr.n_inliers>=min_inliers&&early_exit){early_exit->store(true,std::memory_order_relaxed);fprintf(stderr,"[vm35] mode%d converged (n=%d≥%d)\n",mode,mr.n_inliers,min_inliers);}
    fprintf(stderr,"[vm35] mode%d final: s=%.4f θ=%.2f° n=%d rms=%.3f norm=%.4f\n",mode,mr.s,mr.theta*RADTODEG,mr.n_inliers,mr.rms,mr.norm_score);
    return mr;
}

} // namespace vm35

// ============================================================================
// vm35_solve
// ============================================================================
extern "C" VM35_API int vm35_solve(
    const double* U, int N_img, const double* W, int M,
    const VM35SolveParams* params, VM35SolveResult* result)
{
    using namespace vm35;
    int* saved_mask = result->inlier_mask;
    std::memset(result, 0, sizeof(VM35SolveResult));
    result->inlier_mask = saved_mask;
    result->norm_score = -1.0; result->rms = 1e30;

    // V3.5参数默认值
    if (N_img < 2 || M < 2) { fprintf(stderr, "[vm35] N=%d M=%d too few\n", N_img, M); return -1; }

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
    // 当所有模式n_inliers都很少时(<5)，优先选SNR最高的(θ峰值更可靠)
    // 否则选norm_score最高的(匹配对更多更可靠)
    int any_good = 0;
    for (int m = 0; m < n_modes; ++m)
        if (mres[m].success && mres[m].n_inliers >= 5) { any_good = 1; break; }

    if (any_good) {
        // 正常: 选norm_score最高的
        for (int m = 0; m < n_modes; ++m)
            if (mres[m].success && mres[m].norm_score > best_score)
                { best_score = mres[m].norm_score; best_mode = m; }
    } else {
        // 低匹配: 选SNR最高的(θ峰值最可靠)
        double best_snr = -1.0;
        for (int m = 0; m < n_modes; ++m)
            if (mres[m].success && mres[m].peak_snr > best_snr)
                { best_snr = mres[m].peak_snr; best_mode = m; }
        fprintf(stderr, "[vm35] 低匹配模式: 选SNR最高 mode%d (SNR=%.1fx, n=%d)\n",
                best_mode, best_snr, best_mode >= 0 ? mres[best_mode].n_inliers : 0);
    }

    if (best_mode < 0) { fprintf(stderr, "[vm35] all fail (best=%.4f)\n", best_score); return -1; }

    auto& best = mres[best_mode];

    // V3.5最终版: 移除Phase C全局NN扩充 + Phase D迭代MAD清洗 + Phase D'星点扩增
    // 仅使用Phase B信噪比验证的匹配对直接进入分层拟合
    // Phase B输出的cu/cw是唯一可信的对应关系

    std::vector<double> Wf_fit(M*2);
    apply_flip(W, M, best_mode, Wf_fit.data());

    size_t n_pairs = best.ab.cu.size();
    if (n_pairs < 2) {
        result->s = best.s; result->theta = best.theta;
        result->tx = best.tx; result->ty = best.ty;
        result->n_inliers = best.n_inliers; result->rms = best.rms;
        result->best_mode = best_mode; result->norm_score = best.norm_score;
        result->peak_snr = best.peak_snr; result->n_samples = best.n_samples;
        result->success = 1;
        for (int i = 0; i < N_img; ++i) result->inlier_mask[i] = best.inlier_mask[i];
        double ct0 = std::cos(best.theta), st0 = std::sin(best.theta);
        double cos_dfb = std::cos(params->center_dec * DEGTORAD);
        if (cos_dfb < 1e-10) cos_dfb = 1e-10;
        double s0_s_3600_fb = params->s0 / (best.s * 3600.0);
        bool fx_fb = (best_mode == 1 || best_mode == 3);
        bool fy_fb = (best_mode == 2 || best_mode == 3);
        double sx_fb = fx_fb ? -1.0 : 1.0, sy_fb = fy_fb ? -1.0 : 1.0;
        result->cd[0] = sx_fb * s0_s_3600_fb * ct0 / cos_dfb;
        result->cd[1] = -sx_fb * s0_s_3600_fb * st0 / cos_dfb;
        result->cd[2] = -sy_fb * s0_s_3600_fb * st0;
        result->cd[3] = -sy_fb * s0_s_3600_fb * ct0;
        result->crval[0] = params->center_ra - best.tx / (std::max(cos_dfb, 1e-10) * 3600.0);
        result->crval[1] = params->center_dec - best.ty / 3600.0;
        result->crpix[0] = params->img_width/2; result->crpix[1] = params->img_height/2;
        for (int i = 0; i < 36; ++i) { result->sip_A[i] = 0; result->sip_B[i] = 0; }
        result->debug.theta_snr = best.ab.peak_snr;
        result->debug.theta_peak_deg = best.ab.theta_peak_deg;
        result->debug.best_n_range = best.ab.best_n_range;
        result->debug.median_noise = best.ab.median_noise;
        result->debug.n_phaseb_pairs = best.ab.n_phaseb_pairs;
        result->debug.n_phaseb_corr = best.ab.n_phaseb_corr;
        result->debug.n_phasea_records = best.ab.n_phasea_records;
        result->debug.n_sip_total = 0;
        result->debug.sip_order = 0;
        fprintf(stderr, "[vm35] PhaseB pairs=%zu (<2), fallback linear only\n", n_pairs);
        return 0;
    }

    fprintf(stderr, "[vm35] PhaseB验证匹配对: %zu 对 → 直送分层拟合\n", n_pairs);

    // SIP阶数：BIC在2~4阶自动选最优
    int sip_order = 4;
    // V3.5: 对<30对的场景，跳过SIP用纯仿射（少对时BIC选阶不可靠，SIP容易过拟合）
    if (n_pairs < 30) sip_order = 0;

    // 直接使用Phase B的cu/cw和s/theta调用分层拟合
    auto sip = fit_affine_sip_adaptive(U, Wf_fit.data(), best.ab.cu, best.ab.cw,
        params->s0, params->img_width, params->img_height,
        params->center_ra, params->center_dec, best.s, best.theta, best_mode,
        sip_order, params->wcs_out_path);

    for (int i = 0; i < 36; ++i) {
        result->sip_A[i] = sip.sip_A[i];
        result->sip_B[i] = sip.sip_B[i];
    }
    result->cd[0] = sip.cd[0]; result->cd[1] = sip.cd[1];
    result->cd[2] = sip.cd[2]; result->cd[3] = sip.cd[3];
    result->crval[0] = sip.crval[0]; result->crval[1] = sip.crval[1];
    result->crpix[0] = sip.crpix[0]; result->crpix[1] = sip.crpix[1];
    result->s = sip.s_refined; result->theta = sip.theta_refined;
    result->tx = sip.tx_refined; result->ty = sip.ty_refined;
    result->n_inliers = (int)n_pairs;
    result->rms = sip.rms_px;
    result->best_mode = best_mode;
    result->norm_score = best.norm_score;
    result->peak_snr = best.peak_snr;
    result->n_samples = best.n_samples;
    result->success = 1;
    for (int i = 0; i < N_img; ++i) result->inlier_mask[i] = best.inlier_mask[i];

    result->debug.theta_snr = best.ab.peak_snr;
    result->debug.theta_peak_deg = best.ab.theta_peak_deg;
    result->debug.best_n_range = best.ab.best_n_range;
    result->debug.median_noise = best.ab.median_noise;
    result->debug.n_phaseb_pairs = best.ab.n_phaseb_pairs;
    result->debug.n_phaseb_corr = best.ab.n_phaseb_corr;
    result->debug.n_phasea_records = best.ab.n_phasea_records;
    result->debug.n_phasec_expanded = 0;
    result->debug.n_phased_clean = (int)n_pairs;
    result->debug.n_phased_iterations = 0;
    result->debug.mad_rms_arcsec = 0;
    result->debug.n_expand_mutual = 0;
    result->debug.n_expand_after_filter = 0;
    result->debug.n_sip_total = (int)n_pairs;
    result->debug.sip_order = sip_order;

    fprintf(stderr, "[vm35] OK: mode=%d s=%.6f(Δ=%.4f%%) θ=%.2f° n=%zu SNR=%.1fx "
            "SIP-RMS=%.3fpx SIP-order=%d\n",
            best_mode, sip.s_refined, (sip.s_refined/best.s - 1.0)*100.0,
            sip.theta_refined*RADTODEG, n_pairs, best.peak_snr,
            sip.rms_px, sip_order);
    return 0;
}

extern "C" VM35_API int vm35_count_inliers(
    const double* U, int N_img, const double* W, int M,
    double s, double theta, double tx, double ty,
    double s0, int* inlier_mask, double* out_rms)
{
    using namespace vm35;
    std::vector<double> Wt(M*2);
    apply_similarity(W, M, s, theta, tx, ty, Wt.data());
    auto inl = count_inliers_1to1(U, N_img, Wt.data(), M, 1.0*s0);
    for (int i = 0; i < N_img; ++i) inlier_mask[i] = inl.inlier_mask[i];
    *out_rms = inl.rms;
    return inl.n_inliers;
}
