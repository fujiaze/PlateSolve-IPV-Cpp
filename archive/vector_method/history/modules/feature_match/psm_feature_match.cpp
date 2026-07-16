#include "psm_feature_match.h"
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <stdio.h>
#include <algorithm>
#include <vector>

#define FM_LOG(fmt, ...) fprintf(stderr, "[FM] " fmt "\n", ##__VA_ARGS__)

struct Triangle {
    double ba, ca, a_len;
};

static bool cmp_tri_ba(const Triangle &a, const Triangle &b) {
    return a.ba < b.ba;
}

static void build_triangles_fast(const double *x, const double *y, int n, std::vector<Triangle> &tris) {
    int limit = std::min(n, 100);
    tris.clear();
    tris.reserve(limit * limit * limit / 6);
    
    for (int i = 0; i < limit; i++) {
        for (int j = i + 1; j < limit; j++) {
            for (int k = j + 1; k < limit; k++) {
                double d_ij = sqrt((x[i]-x[j])*(x[i]-x[j]) + (y[i]-y[j])*(y[i]-y[j]));
                double d_ik = sqrt((x[i]-x[k])*(x[i]-x[k]) + (y[i]-y[k])*(y[i]-y[k]));
                double d_jk = sqrt((x[j]-x[k])*(x[j]-x[k]) + (y[j]-y[k])*(y[j]-y[k]));
                
                double sides[3] = {d_ij, d_ik, d_jk};
                std::sort(sides, sides + 3, std::greater<double>());
                double a = sides[0], b = sides[1], c = sides[2];
                
                if (a < 10.0) continue;
                
                double ba = b / a;
                double ca = c / a;
                
                if (ba > 0.92) continue;
                
                tris.push_back({ba, ca, a});
            }
        }
    }
    
    std::sort(tris.begin(), tris.end(), cmp_tri_ba);
}

static int match_triangles_count(const std::vector<Triangle> &tris1, 
                                  const std::vector<Triangle> &tris2, 
                                  double radius) {
    int count = 0;
    int n1 = tris1.size();
    
    for (const auto &t2 : tris2) {
        double ba2 = t2.ba;
        double ca2 = t2.ca;
        
        int lo = 0, hi = n1;
        while (lo < hi) {
            int mid = (lo + hi) / 2;
            if (tris1[mid].ba < ba2 - radius)
                lo = mid + 1;
            else
                hi = mid;
        }
        
        for (int i = lo; i < n1; i++) {
            if (tris1[i].ba > ba2 + radius) break;
            
            double dist = sqrt((tris1[i].ba - ba2)*(tris1[i].ba - ba2) + 
                              (tris1[i].ca - ca2)*(tris1[i].ca - ca2));
            if (dist < radius) count++;
        }
    }
    
    return count;
}

static void lstsq_affine(const double *img_x, const double *img_y, 
                         const double *cat_x, const double *cat_y, 
                         int n,
                         double &a0, double &a1, double &a2,
                         double &b0, double &b1, double &b2) {
    double S1 = n;
    double Sx = 0, Sy = 0, Sxx = 0, Syy = 0, Sxy = 0;
    double Sxa = 0, Sya = 0, Sxb = 0, Syb = 0;
    
    for (int i = 0; i < n; i++) {
        double xi = img_x[i], yi = img_y[i];
        double xa = cat_x[i], yb = cat_y[i];
        
        Sx += xi; Sy += yi;
        Sxx += xi * xi; Syy += yi * yi; Sxy += xi * yi;
        Sxa += xi * xa; Sya += yi * xa;
        Sxb += xi * yb; Syb += yi * yb;
    }
    
    double A[3][3] = {
        {S1, Sx, Sy},
        {Sx, Sxx, Sxy},
        {Sy, Sxy, Syy}
    };
    
    double bx[3] = {0, Sxa, Sya};
    double by[3] = {0, Sxb, Syb};
    
    for (int i = 0; i < n; i++) {
        bx[0] += cat_x[i];
        by[0] += cat_y[i];
    }
    
    double det = A[0][0] * (A[1][1]*A[2][2] - A[1][2]*A[2][1])
              - A[0][1] * (A[1][0]*A[2][2] - A[1][2]*A[2][0])
              + A[0][2] * (A[1][0]*A[2][1] - A[1][1]*A[2][0]);
    
    if (fabs(det) < 1e-15) {
        a0 = a1 = a2 = b0 = b1 = b2 = 0;
        return;
    }
    
    double inv[3][3];
    inv[0][0] = (A[1][1]*A[2][2] - A[1][2]*A[2][1]) / det;
    inv[0][1] = (A[0][2]*A[2][1] - A[0][1]*A[2][2]) / det;
    inv[0][2] = (A[0][1]*A[1][2] - A[0][2]*A[1][1]) / det;
    inv[1][0] = (A[1][2]*A[2][0] - A[1][0]*A[2][2]) / det;
    inv[1][1] = (A[0][0]*A[2][2] - A[0][2]*A[2][0]) / det;
    inv[1][2] = (A[0][2]*A[1][0] - A[0][0]*A[1][2]) / det;
    inv[2][0] = (A[1][0]*A[2][1] - A[1][1]*A[2][0]) / det;
    inv[2][1] = (A[0][1]*A[2][0] - A[0][0]*A[2][1]) / det;
    inv[2][2] = (A[0][0]*A[1][1] - A[0][1]*A[1][0]) / det;
    
    a0 = inv[0][0]*bx[0] + inv[0][1]*bx[1] + inv[0][2]*bx[2];
    a1 = inv[1][0]*bx[0] + inv[1][1]*bx[1] + inv[1][2]*bx[2];
    a2 = inv[2][0]*bx[0] + inv[2][1]*bx[1] + inv[2][2]*bx[2];
    
    b0 = inv[0][0]*by[0] + inv[0][1]*by[1] + inv[0][2]*by[2];
    b1 = inv[1][0]*by[0] + inv[1][1]*by[1] + inv[1][2]*by[2];
    b2 = inv[2][0]*by[0] + inv[2][1]*by[1] + inv[2][2]*by[2];
}

static int kdtree_match(const double *img_x, const double *img_y, int n_img,
                        const double *cat_x, const double *cat_y, int n_cat,
                        double max_dist,
                        int *matched_idx, double *matched_dist) {
    int n_matched = 0;
    
    for (int i = 0; i < n_img; i++) {
        double tx = img_x[i];
        double ty = img_y[i];
        
        double min_dist = 1e30;
        int min_j = -1;
        
        for (int j = 0; j < n_cat; j++) {
            double dx = tx - cat_x[j];
            double dy = ty - cat_y[j];
            double d = dx*dx + dy*dy;
            if (d < min_dist) {
                min_dist = d;
                min_j = j;
            }
        }
        
        min_dist = sqrt(min_dist);
        if (min_dist < max_dist) {
            matched_idx[i] = min_j;
            matched_dist[i] = min_dist;
            n_matched++;
        } else {
            matched_idx[i] = -1;
        }
    }
    
    return n_matched;
}

int psm_direct_align(
    const double *img_x, const double *img_y, int n_img,
    const double *cat_x, const double *cat_y, int n_cat,
    const PSMDirectAlignConfig *config,
    PSMDirectAlignResult *out_result) {
    
    if (!img_x || !img_y || n_img <= 0 || !cat_x || !cat_y || n_cat <= 0 || !config || !out_result) {
        return PSM_ERR_INVALID_PARAM;
    }
    
    memset(out_result, 0, sizeof(PSMDirectAlignResult));
    
    int n_bright = config->n_bright > 0 ? config->n_bright : 100;
    double max_dist = config->max_dist > 0 ? config->max_dist : 25.0;
    int max_iter = config->max_iterations > 0 ? config->max_iterations : 5;
    
    int *matched_idx = (int *)malloc(n_img * sizeof(int));
    double *matched_dist = (double *)malloc(n_img * sizeof(double));
    double *tx = (double *)malloc(n_img * sizeof(double));
    double *ty = (double *)malloc(n_img * sizeof(double));
    
    double best_score = 0;
    int best_flip_x = 0, best_flip_y = 0;
    double best_a0, best_a1, best_a2, best_b0, best_b1, best_b2;
    int best_n_matched = 0;
    double best_mean_dist = 1e30;
    
    for (int flip_x = 0; flip_x <= 1; flip_x++) {
        for (int flip_y = 0; flip_y <= 1; flip_y++) {
            std::vector<Triangle> img_tris, cat_tris;
            
            std::vector<double> fx(n_img), fy(n_img);
            for (int i = 0; i < n_img; i++) {
                fx[i] = flip_x ? -img_x[i] : img_x[i];
                fy[i] = flip_y ? -img_y[i] : img_y[i];
            }
            
            build_triangles_fast(fx.data(), fy.data(), std::min(n_img, n_bright), img_tris);
            build_triangles_fast(cat_x, cat_y, std::min(n_cat, n_bright), cat_tris);
            
            int tri_count = match_triangles_count(img_tris, cat_tris, 0.0008);
            
            if (tri_count < 100) continue;
            
            double img_cx = 0, img_cy = 0, cat_cx = 0, cat_cy = 0;
            int nc = std::min(n_img, n_bright);
            for (int i = 0; i < nc; i++) {
                img_cx += fx[i]; img_cy += fy[i];
            }
            img_cx /= nc; img_cy /= nc;
            
            nc = std::min(n_cat, n_bright);
            for (int i = 0; i < nc; i++) {
                cat_cx += cat_x[i]; cat_cy += cat_y[i];
            }
            cat_cx /= nc; cat_cy /= nc;
            
            double dx = cat_cx - img_cx;
            double dy = cat_cy - img_cy;
            
            double a0 = dx, a1 = 1, a2 = 0;
            double b0 = dy, b1 = 0, b2 = 1;
            
            for (int iter = 0; iter < max_iter; iter++) {
                for (int i = 0; i < n_img; i++) {
                    tx[i] = a0 + a1*fx[i] + a2*fy[i];
                    ty[i] = b0 + b1*fx[i] + b2*fy[i];
                }
                
                int n_m = kdtree_match(tx, ty, n_img, cat_x, cat_y, n_cat, max_dist, matched_idx, matched_dist);
                
                if (n_m < 10) break;
                
                std::vector<double> mx(n_m), my(n_m), cx(n_m), cy(n_m);
                int idx = 0;
                for (int i = 0; i < n_img; i++) {
                    if (matched_idx[i] >= 0) {
                        mx[idx] = fx[i];
                        my[idx] = fy[i];
                        cx[idx] = cat_x[matched_idx[i]];
                        cy[idx] = cat_y[matched_idx[i]];
                        idx++;
                    }
                }
                
                lstsq_affine(mx.data(), my.data(), cx.data(), cy.data(), n_m, a0, a1, a2, b0, b1, b2);
            }
            
            for (int i = 0; i < n_img; i++) {
                tx[i] = a0 + a1*fx[i] + a2*fy[i];
                ty[i] = b0 + b1*fx[i] + b2*fy[i];
            }
            
            int n_final = kdtree_match(tx, ty, n_img, cat_x, cat_y, n_cat, max_dist, matched_idx, matched_dist);
            
            double mean_dist = 0;
            for (int i = 0; i < n_img; i++) {
                if (matched_idx[i] >= 0) mean_dist += matched_dist[i];
            }
            mean_dist /= n_final;
            
            double score = n_final / (1 + mean_dist / 10);
            
            if (score > best_score) {
                best_score = score;
                best_flip_x = flip_x;
                best_flip_y = flip_y;
                best_a0 = a0; best_a1 = a1; best_a2 = a2;
                best_b0 = b0; best_b1 = b1; best_b2 = b2;
                best_n_matched = n_final;
                best_mean_dist = mean_dist;
            }
        }
    }
    
    free(matched_idx);
    free(matched_dist);
    free(tx);
    free(ty);
    
    if (best_n_matched == 0) {
        return PSM_ERR_NO_MATCH;
    }
    
    out_result->pair_count = best_n_matched;
    out_result->mean_dist = best_mean_dist;
    out_result->a0 = best_a0; out_result->a1 = best_a1; out_result->a2 = best_a2;
    out_result->b0 = best_b0; out_result->b1 = best_b1; out_result->b2 = best_b2;
    out_result->flip_x = best_flip_x;
    out_result->flip_y = best_flip_y;
    
    FM_LOG("Direct align: %d matches, mean_dist=%.2fpx, flip=(%d,%d)", 
           best_n_matched, best_mean_dist, best_flip_x, best_flip_y);
    
    return PSM_OK;
}

void psm_free_direct_align_result(PSMDirectAlignResult *result) {
    if (result && result->pairs) {
        free(result->pairs);
        result->pairs = NULL;
        result->pair_count = 0;
    }
}
