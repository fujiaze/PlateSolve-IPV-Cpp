#include "psm_sip.h"
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <stdio.h>
#include <algorithm>
#include <vector>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

#define SIP_LOG(fmt, ...) fprintf(stderr, "[SIP] " fmt "\n", ##__VA_ARGS__)

void sip_trans_init(SIP_Transform *t, int order)
{
    memset(t, 0, sizeof(SIP_Transform));
    t->order = order;
    t->x10 = 1.0;
    t->y01 = 1.0;
}

void sip_trans_copy(SIP_Transform *dst, const SIP_Transform *src)
{
    memcpy(dst, src, sizeof(SIP_Transform));
}

void sip_trans_apply(const SIP_Transform *t, double x, double y, double *out_x, double *out_y)
{
    double x2 = x * x, y2 = y * y;
    double x3 = x2 * x, y3 = y2 * y;
    double x4 = x3 * x, y4 = y3 * y;
    double x5 = x4 * x, y5 = y4 * y;
    double xy = x * y;
    double x2y = x2 * y, xy2 = x * y2;
    double x3y = x3 * y, x2y2 = x2 * y2, xy3 = x * y3;
    double x4y = x4 * y, x3y2 = x3 * y2, x2y3 = x2 * y3, xy4 = x * y4;
    
    *out_x = t->x00 + t->x10 * x + t->x01 * y
           + t->x20 * x2 + t->x11 * xy + t->x02 * y2
           + t->x30 * x3 + t->x21 * x2y + t->x12 * xy2 + t->x03 * y3
           + t->x40 * x4 + t->x31 * x3y + t->x22 * x2y2 + t->x13 * xy3 + t->x04 * y4
           + t->x50 * x5 + t->x41 * x4y + t->x32 * x3y2 + t->x23 * x2y3 + t->x14 * xy4 + t->x05 * y5;
    
    *out_y = t->y00 + t->y10 * x + t->y01 * y
           + t->y20 * x2 + t->y11 * xy + t->y02 * y2
           + t->y30 * x3 + t->y21 * x2y + t->y12 * xy2 + t->y03 * y3
           + t->y40 * x4 + t->y31 * x3y + t->y22 * x2y2 + t->y13 * xy3 + t->y04 * y4
           + t->y50 * x5 + t->y41 * x4y + t->y32 * x3y2 + t->y23 * x2y3 + t->y14 * xy4 + t->y05 * y5;
}

void sip_trans_apply_array(const SIP_Transform *t, SIP_Point *points, int n)
{
    for (int i = 0; i < n; i++) {
        double x = points[i].x, y = points[i].y;
        sip_trans_apply(t, x, y, &points[i].x, &points[i].y);
    }
}

double sip_trans_det(const SIP_Transform *t)
{
    return t->x10 * t->y01 - t->y10 * t->x01;
}

void sip_trans_get_cd(const SIP_Transform *t, double cd[2][2])
{
    cd[0][0] = t->x10;
    cd[0][1] = t->x01;
    cd[1][0] = t->y10;
    cd[1][1] = t->y01;
}

void sip_trans_invert_cd(const SIP_Transform *t, double cd_inv[2][2])
{
    double det = sip_trans_det(t);
    double invdet = 1.0 / det;
    cd_inv[0][0] =  invdet * t->y01;
    cd_inv[0][1] = -invdet * t->x01;
    cd_inv[1][0] = -invdet * t->y10;
    cd_inv[1][1] =  invdet * t->x10;
}

SIP_Point *sip_create_grid(int rx, int ry, int n_points)
{
    int total = n_points * n_points;
    SIP_Point *grid = (SIP_Point *)malloc(total * sizeof(SIP_Point));
    
    double step_x = rx / (double)(n_points - 1);
    double step_y = ry / (double)(n_points - 1);
    double offset_x = -rx / 2.0;
    double offset_y = -ry / 2.0;
    
    int idx = 0;
    for (int j = 0; j < n_points; j++) {
        for (int i = 0; i < n_points; i++) {
            grid[idx].x = offset_x + i * step_x;
            grid[idx].y = offset_y + j * step_y;
            idx++;
        }
    }
    return grid;
}

void sip_free_grid(SIP_Point *grid)
{
    free(grid);
}

static int sip_nbasis(int order)
{
    return (order + 1) * (order + 2) / 2;
}

static void sip_build_basis(double x, double y, double *basis, int order)
{
    int idx = 0;
    for (int i = 0; i <= order; i++) {
        for (int j = 0; j <= order - i; j++) {
            basis[idx] = pow(x, i) * pow(y, j);
            idx++;
        }
    }
}

static void sip_set_coeffs(SIP_Transform *t, const double *cx, const double *cy, int order)
{
    t->order = order;
    t->x00 = cx[0]; t->y00 = cy[0];
    t->x10 = cx[1]; t->y10 = cy[1];
    t->x01 = cx[2]; t->y01 = cy[2];
    if (order >= 2) {
        t->x20 = cx[3]; t->y20 = cy[3];
        t->x11 = cx[4]; t->y11 = cy[4];
        t->x02 = cx[5]; t->y02 = cy[5];
    }
    if (order >= 3) {
        t->x30 = cx[6]; t->y30 = cy[6];
        t->x21 = cx[7]; t->y21 = cy[7];
        t->x12 = cx[8]; t->y12 = cy[8];
        t->x03 = cx[9]; t->y03 = cy[9];
    }
    if (order >= 4) {
        t->x40 = cx[10]; t->y40 = cy[10];
        t->x31 = cx[11]; t->y31 = cy[11];
        t->x22 = cx[12]; t->y22 = cy[12];
        t->x13 = cx[13]; t->y13 = cy[13];
        t->x04 = cx[14]; t->y04 = cy[14];
    }
    if (order >= 5) {
        t->x50 = cx[15]; t->y50 = cy[15];
        t->x41 = cx[16]; t->y41 = cy[16];
        t->x32 = cx[17]; t->y32 = cy[17];
        t->x23 = cx[18]; t->y23 = cy[18];
        t->x14 = cx[19]; t->y14 = cy[19];
        t->x05 = cx[20]; t->y05 = cy[20];
    }
}

int sip_trans_fit(SIP_Transform *t, const SIP_Point *src, const SIP_Point *dst, int n,
                  int max_iter, double tolerance)
{
    int nbasis = sip_nbasis(t->order);
    if (n < nbasis) return -1;
    
    std::vector<double> XtX(nbasis * nbasis, 0);
    std::vector<double> Xtdx(nbasis, 0), Xtdy(nbasis, 0);
    std::vector<double> basis(nbasis);
    
    for (int k = 0; k < n; k++) {
        sip_build_basis(src[k].x, src[k].y, basis.data(), t->order);
        double dx = dst[k].x;
        double dy = dst[k].y;
        
        for (int i = 0; i < nbasis; i++) {
            for (int j = 0; j < nbasis; j++) {
                XtX[i * nbasis + j] += basis[i] * basis[j];
            }
            Xtdx[i] += basis[i] * dx;
            Xtdy[i] += basis[i] * dy;
        }
    }
    
    std::vector<double> cx(nbasis, 0), cy(nbasis, 0);
    
    for (int iter = 0; iter < max_iter; iter++) {
        double max_change = 0;
        for (int i = 0; i < nbasis; i++) {
            double diag = XtX[i * nbasis + i];
            if (diag < 1e-10) diag = 1e-10;
            
            double sum_x = Xtdx[i], sum_y = Xtdy[i];
            for (int j = 0; j < nbasis; j++) {
                if (j != i) {
                    sum_x -= XtX[i * nbasis + j] * cx[j];
                    sum_y -= XtX[i * nbasis + j] * cy[j];
                }
            }
            double new_cx = sum_x / diag;
            double new_cy = sum_y / diag;
            
            max_change = std::max(max_change, fabs(new_cx - cx[i]));
            max_change = std::max(max_change, fabs(new_cy - cy[i]));
            cx[i] = new_cx;
            cy[i] = new_cy;
        }
        if (max_change < tolerance) break;
    }
    
    sip_set_coeffs(t, cx.data(), cy.data(), t->order);
    
    double sum_resid2 = 0;
    for (int k = 0; k < n; k++) {
        double px, py;
        sip_trans_apply(t, src[k].x, src[k].y, &px, &py);
        double rx = dst[k].x - px;
        double ry = dst[k].y - py;
        sum_resid2 += rx * rx + ry * ry;
    }
    t->rms = sqrt(sum_resid2 / n);
    t->match_count = n;
    
    return 0;
}

static void sip_mv_decomp(double m[2][2], double v0, double v1, double *r0, double *r1)
{
    *r0 = m[0][0] * v0 + m[0][1] * v1;
    *r1 = m[1][0] * v0 + m[1][1] * v1;
}

static void sip_scale_coeffs(SIP_Transform *t, double scale)
{
    t->x20 /= scale * scale;
    t->x11 /= scale * scale;
    t->x02 /= scale * scale;
    t->y20 /= scale * scale;
    t->y11 /= scale * scale;
    t->y02 /= scale * scale;
    
    t->x30 /= scale * scale * scale;
    t->x21 /= scale * scale * scale;
    t->x12 /= scale * scale * scale;
    t->x03 /= scale * scale * scale;
    t->y30 /= scale * scale * scale;
    t->y21 /= scale * scale * scale;
    t->y12 /= scale * scale * scale;
    t->y03 /= scale * scale * scale;
    
    t->x40 /= scale * scale * scale * scale;
    t->x31 /= scale * scale * scale * scale;
    t->x22 /= scale * scale * scale * scale;
    t->x13 /= scale * scale * scale * scale;
    t->x04 /= scale * scale * scale * scale;
    t->y40 /= scale * scale * scale * scale;
    t->y31 /= scale * scale * scale * scale;
    t->y22 /= scale * scale * scale * scale;
    t->y13 /= scale * scale * scale * scale;
    t->y04 /= scale * scale * scale * scale;
    
    t->x50 /= scale * scale * scale * scale * scale;
    t->x41 /= scale * scale * scale * scale * scale;
    t->x32 /= scale * scale * scale * scale * scale;
    t->x23 /= scale * scale * scale * scale * scale;
    t->x14 /= scale * scale * scale * scale * scale;
    t->x05 /= scale * scale * scale * scale * scale;
    t->y50 /= scale * scale * scale * scale * scale;
    t->y41 /= scale * scale * scale * scale * scale;
    t->y32 /= scale * scale * scale * scale * scale;
    t->y23 /= scale * scale * scale * scale * scale;
    t->y14 /= scale * scale * scale * scale * scale;
    t->y05 /= scale * scale * scale * scale * scale;
}

int sip_coeffs_compute(const SIP_Transform *t, int rx, int ry, SIP_Coeffs *sip)
{
    SIP_Transform trans_work;
    sip_trans_copy(&trans_work, t);
    
    trans_work.x00 = 0.0;
    trans_work.y00 = 0.0;
    
    int nbpoints = SIP_GRID_SIZE * SIP_GRID_SIZE;
    SIP_Point *uvgrid = sip_create_grid(rx, ry, SIP_GRID_SIZE);
    SIP_Point *xygrid = sip_create_grid(rx, ry, SIP_GRID_SIZE);
    
    sip_trans_apply_array(&trans_work, xygrid, nbpoints);
    
    double cd[2][2], cd_inv[2][2];
    sip_trans_get_cd(&trans_work, cd);
    sip_trans_invert_cd(&trans_work, cd_inv);
    
    SIP_Transform transUV;
    sip_trans_init(&transUV, 1);
    transUV.x10 = cd_inv[0][0];
    transUV.x01 = cd_inv[0][1];
    transUV.y10 = cd_inv[1][0];
    transUV.y01 = cd_inv[1][1];
    transUV.x00 = 0;
    transUV.y00 = 0;
    
    sip_trans_apply_array(&transUV, xygrid, nbpoints);
    
    SIP_Transform revtrans;
    sip_trans_init(&revtrans, trans_work.order);
    
    int status = sip_trans_fit(&revtrans, xygrid, uvgrid, nbpoints, 50, 1e-10);
    if (status) {
        SIP_LOG("Failed to compute inverse SIP coefficients");
        sip_free_grid(uvgrid);
        sip_free_grid(xygrid);
        return -1;
    }
    
    int N = trans_work.order;
    memset(sip, 0, sizeof(SIP_Coeffs));
    sip->order = N;
    
    sip_mv_decomp(cd_inv, trans_work.x20, trans_work.y20, &sip->A[2][0], &sip->B[2][0]);
    sip_mv_decomp(cd_inv, trans_work.x11, trans_work.y11, &sip->A[1][1], &sip->B[1][1]);
    sip_mv_decomp(cd_inv, trans_work.x02, trans_work.y02, &sip->A[0][2], &sip->B[0][2]);
    
    sip->AP[0][0] = revtrans.x00;
    sip->AP[1][0] = revtrans.x10 - 1.0;
    sip->AP[0][1] = revtrans.x01;
    sip->AP[2][0] = revtrans.x20;
    sip->AP[1][1] = revtrans.x11;
    sip->AP[0][2] = revtrans.x02;
    
    sip->BP[0][0] = revtrans.y00;
    sip->BP[1][0] = revtrans.y10;
    sip->BP[0][1] = revtrans.y01 - 1.0;
    sip->BP[2][0] = revtrans.y20;
    sip->BP[1][1] = revtrans.y11;
    sip->BP[0][2] = revtrans.y02;
    
    if (N >= 3) {
        sip_mv_decomp(cd_inv, trans_work.x30, trans_work.y30, &sip->A[3][0], &sip->B[3][0]);
        sip_mv_decomp(cd_inv, trans_work.x21, trans_work.y21, &sip->A[2][1], &sip->B[2][1]);
        sip_mv_decomp(cd_inv, trans_work.x12, trans_work.y12, &sip->A[1][2], &sip->B[1][2]);
        sip_mv_decomp(cd_inv, trans_work.x03, trans_work.y03, &sip->A[0][3], &sip->B[0][3]);
        
        sip->AP[3][0] = revtrans.x30;
        sip->AP[2][1] = revtrans.x21;
        sip->AP[1][2] = revtrans.x12;
        sip->AP[0][3] = revtrans.x03;
        
        sip->BP[3][0] = revtrans.y30;
        sip->BP[2][1] = revtrans.y21;
        sip->BP[1][2] = revtrans.y12;
        sip->BP[0][3] = revtrans.y03;
    }
    
    if (N >= 4) {
        sip_mv_decomp(cd_inv, trans_work.x40, trans_work.y40, &sip->A[4][0], &sip->B[4][0]);
        sip_mv_decomp(cd_inv, trans_work.x31, trans_work.y31, &sip->A[3][1], &sip->B[3][1]);
        sip_mv_decomp(cd_inv, trans_work.x22, trans_work.y22, &sip->A[2][2], &sip->B[2][2]);
        sip_mv_decomp(cd_inv, trans_work.x13, trans_work.y13, &sip->A[1][3], &sip->B[1][3]);
        sip_mv_decomp(cd_inv, trans_work.x04, trans_work.y04, &sip->A[0][4], &sip->B[0][4]);
        
        sip->AP[4][0] = revtrans.x40;
        sip->AP[3][1] = revtrans.x31;
        sip->AP[2][2] = revtrans.x22;
        sip->AP[1][3] = revtrans.x13;
        sip->AP[0][4] = revtrans.x04;
        
        sip->BP[4][0] = revtrans.y40;
        sip->BP[3][1] = revtrans.y31;
        sip->BP[2][2] = revtrans.y22;
        sip->BP[1][3] = revtrans.y13;
        sip->BP[0][4] = revtrans.y04;
    }
    
    if (N >= 5) {
        sip_mv_decomp(cd_inv, trans_work.x50, trans_work.y50, &sip->A[5][0], &sip->B[5][0]);
        sip_mv_decomp(cd_inv, trans_work.x41, trans_work.y41, &sip->A[4][1], &sip->B[4][1]);
        sip_mv_decomp(cd_inv, trans_work.x32, trans_work.y32, &sip->A[3][2], &sip->B[3][2]);
        sip_mv_decomp(cd_inv, trans_work.x23, trans_work.y23, &sip->A[2][3], &sip->B[2][3]);
        sip_mv_decomp(cd_inv, trans_work.x14, trans_work.y14, &sip->A[1][4], &sip->B[1][4]);
        sip_mv_decomp(cd_inv, trans_work.x05, trans_work.y05, &sip->A[0][5], &sip->B[0][5]);
        
        sip->AP[5][0] = revtrans.x50;
        sip->AP[4][1] = revtrans.x41;
        sip->AP[3][2] = revtrans.x32;
        sip->AP[2][3] = revtrans.x23;
        sip->AP[1][4] = revtrans.x14;
        sip->AP[0][5] = revtrans.x05;
        
        sip->BP[5][0] = revtrans.y50;
        sip->BP[4][1] = revtrans.y41;
        sip->BP[3][2] = revtrans.y32;
        sip->BP[2][3] = revtrans.y23;
        sip->BP[1][4] = revtrans.y14;
        sip->BP[0][5] = revtrans.y05;
    }
    
    sip->valid = 1;
    
    sip_free_grid(uvgrid);
    sip_free_grid(xygrid);
    
    return 0;
}

void sip_apply_forward(const SIP_Coeffs *sip, double x, double y, double *out_x, double *out_y)
{
    double dx = 0, dy = 0;
    int N = sip->order;
    
    for (int i = 0; i <= N; i++) {
        for (int j = 0; j <= N - i; j++) {
            if (i == 0 && j == 0) continue;
            if (i == 1 && j == 0) continue;
            if (i == 0 && j == 1) continue;
            double term = pow(x, i) * pow(y, j);
            dx += sip->A[i][j] * term;
            dy += sip->B[i][j] * term;
        }
    }
    
    *out_x = x + dx;
    *out_y = y + dy;
}

void sip_apply_inverse(const SIP_Coeffs *sip, double u, double v, double *out_x, double *out_y)
{
    double du = 0, dv = 0;
    int N = sip->order;
    
    for (int i = 0; i <= N; i++) {
        for (int j = 0; j <= N - i; j++) {
            if (i == 0 && j == 0) continue;
            if (i == 1 && j == 0) continue;
            if (i == 0 && j == 1) continue;
            double term = pow(u, i) * pow(v, j);
            du += sip->AP[i][j] * term;
            dv += sip->BP[i][j] * term;
        }
    }
    
    *out_x = u + du;
    *out_y = v + dv;
}

static void sip_fit_linear(const double *src_x, const double *src_y,
                           const double *dst_x, const double *dst_y, int n,
                           SIP_Transform *t)
{
    double sum_xx = 0, sum_yy = 0, sum_xy = 0;
    double sum_xdx = 0, sum_xdy = 0, sum_ydx = 0, sum_ydy = 0;
    
    for (int i = 0; i < n; i++) {
        double sx = src_x[i], sy = src_y[i];
        double dx = dst_x[i], dy = dst_y[i];
        sum_xx += sx * sx;
        sum_yy += sy * sy;
        sum_xy += sx * sy;
        sum_xdx += sx * dx;
        sum_xdy += sx * dy;
        sum_ydx += sy * dx;
        sum_ydy += sy * dy;
    }
    
    double A[4] = {sum_xx, sum_xy, sum_xy, sum_yy};
    double bx[2] = {sum_xdx, sum_ydx};
    double by[2] = {sum_xdy, sum_ydy};
    
    double det = A[0] * A[3] - A[1] * A[2];
    if (fabs(det) < 1e-10) {
        sip_trans_init(t, 1);
        return;
    }
    
    double x10 = (A[3] * bx[0] - A[1] * bx[1]) / det;
    double x01 = (-A[2] * bx[0] + A[0] * bx[1]) / det;
    double y10 = (A[3] * by[0] - A[1] * by[1]) / det;
    double y01 = (-A[2] * by[0] + A[0] * by[1]) / det;
    
    double x00 = 0, y00 = 0;
    for (int i = 0; i < n; i++) {
        x00 += dst_x[i] - x10 * src_x[i] - x01 * src_y[i];
        y00 += dst_y[i] - y10 * src_x[i] - y01 * src_y[i];
    }
    x00 /= n;
    y00 /= n;
    
    sip_trans_init(t, 1);
    t->x00 = x00;
    t->y00 = y00;
    t->x10 = x10;
    t->x01 = x01;
    t->y10 = y10;
    t->y01 = y01;
}

int sip_fit_refine(
    const double *img_x, const double *img_y, int n_img,
    const double *cat_x, const double *cat_y, int n_cat,
    double center_x, double center_y,
    int img_w, int img_h,
    int sip_order,
    SIP_Transform *out_trans,
    SIP_Coeffs *out_sip,
    double *out_rms)
{
    SIP_LOG("=== SIP Fitting (order %d) ===", sip_order);
    SIP_LOG("  Image stars: %d, Catalog stars: %d", n_img, n_cat);
    SIP_LOG("  Image size: %d x %d", img_w, img_h);
    
    std::vector<double> src_x(n_img), src_y(n_img);
    std::vector<double> dst_x(n_img), dst_y(n_img);
    
    for (int i = 0; i < n_img; i++) {
        src_x[i] = img_x[i] - center_x;
        src_y[i] = img_y[i] - center_y;
        dst_x[i] = cat_x[i] - center_x;
        dst_y[i] = cat_y[i] - center_y;
    }
    
    double max_radius = sqrt((img_w/2.0)*(img_w/2.0) + (img_h/2.0)*(img_h/2.0));
    SIP_LOG("  Max radius for normalization: %.1f px", max_radius);
    
    SIP_Transform trans;
    sip_fit_linear(src_x.data(), src_y.data(), dst_x.data(), dst_y.data(), n_img, &trans);
    
    SIP_LOG("  Linear fit: x00=%.4f x10=%.6f x01=%.6f", trans.x00, trans.x10, trans.x01);
    SIP_LOG("              y00=%.4f y10=%.6f y01=%.6f", trans.y00, trans.y10, trans.y01);
    
    if (sip_order > 1) {
        SIP_Transform trans_full;
        sip_trans_init(&trans_full, sip_order);
        
        double x_center = 0, y_center = 0;
        for (int i = 0; i < n_img; i++) {
            x_center += src_x[i];
            y_center += src_y[i];
        }
        x_center /= n_img;
        y_center /= n_img;
        
        double x_scale = 0, y_scale = 0;
        for (int i = 0; i < n_img; i++) {
            double dx = fabs(src_x[i] - x_center);
            double dy = fabs(src_y[i] - y_center);
            if (dx > x_scale) x_scale = dx;
            if (dy > y_scale) y_scale = dy;
        }
        if (x_scale < 1) x_scale = 1;
        if (y_scale < 1) y_scale = 1;
        
        int nbasis = sip_nbasis(sip_order);
        std::vector<double> XtX(nbasis * nbasis, 0);
        std::vector<double> Xtdx(nbasis, 0), Xtdy(nbasis, 0);
        std::vector<double> basis(nbasis);
        
        for (int k = 0; k < n_img; k++) {
            double u = (src_x[k] - x_center) / x_scale;
            double v = (src_y[k] - y_center) / y_scale;
            sip_build_basis(u, v, basis.data(), sip_order);
            double dx = dst_x[k];
            double dy = dst_y[k];
            
            for (int i = 0; i < nbasis; i++) {
                for (int j = 0; j < nbasis; j++) {
                    XtX[i * nbasis + j] += basis[i] * basis[j];
                }
                Xtdx[i] += basis[i] * dx;
                Xtdy[i] += basis[i] * dy;
            }
        }
        
        std::vector<double> A(nbasis * nbasis);
        for (int i = 0; i < nbasis * nbasis; i++) A[i] = XtX[i];
        std::vector<double> bx(nbasis), by(nbasis);
        for (int i = 0; i < nbasis; i++) { bx[i] = Xtdx[i]; by[i] = Xtdy[i]; }
        
        std::vector<double> biggest_val(nbasis, 0);
        for (int i = 0; i < nbasis; i++) {
            for (int j = 0; j < nbasis; j++) {
                if (fabs(A[i * nbasis + j]) > biggest_val[i]) {
                    biggest_val[i] = fabs(A[i * nbasis + j]);
                }
            }
            if (biggest_val[i] < 1e-15) biggest_val[i] = 1.0;
        }
        
        double maxtol = pow(1e-12, sip_order);
        
        for (int col = 0; col < nbasis - 1; col++) {
            int max_row = col;
            double max_ratio = fabs(A[col * nbasis + col]) / biggest_val[col];
            for (int row = col + 1; row < nbasis; row++) {
                double ratio = fabs(A[row * nbasis + col]) / biggest_val[row];
                if (ratio > max_ratio) {
                    max_ratio = ratio;
                    max_row = row;
                }
            }
            
            if (max_row != col) {
                for (int j = 0; j < nbasis; j++) {
                    std::swap(A[col * nbasis + j], A[max_row * nbasis + j]);
                }
                std::swap(bx[col], bx[max_row]);
                std::swap(by[col], by[max_row]);
                std::swap(biggest_val[col], biggest_val[max_row]);
            }
            
            double diag = A[col * nbasis + col];
            if (fabs(diag / biggest_val[col]) < maxtol) continue;
            
            for (int row = col + 1; row < nbasis; row++) {
                double factor = A[row * nbasis + col] / diag;
                A[row * nbasis + col] = 0;
                for (int j = col + 1; j < nbasis; j++) {
                    A[row * nbasis + j] -= factor * A[col * nbasis + j];
                }
                bx[row] -= factor * bx[col];
                by[row] -= factor * by[col];
            }
        }
        
        std::vector<double> cx(nbasis, 0), cy(nbasis, 0);
        double last_diag = A[(nbasis - 1) * nbasis + (nbasis - 1)];
        if (fabs(last_diag / biggest_val[nbasis - 1]) >= maxtol) {
            cx[nbasis - 1] = bx[nbasis - 1] / last_diag;
            cy[nbasis - 1] = by[nbasis - 1] / last_diag;
        }
        
        for (int i = nbasis - 2; i >= 0; i--) {
            double sum_x = 0, sum_y = 0;
            for (int j = i + 1; j < nbasis; j++) {
                sum_x += A[i * nbasis + j] * cx[j];
                sum_y += A[i * nbasis + j] * cy[j];
            }
            double diag = A[i * nbasis + i];
            if (fabs(diag / biggest_val[i]) < maxtol) {
                cx[i] = 0;
                cy[i] = 0;
            } else {
                cx[i] = (bx[i] - sum_x) / diag;
                cy[i] = (by[i] - sum_y) / diag;
            }
        }
        
        double sum_rms = 0;
        for (int i = 0; i < n_img; i++) {
            double u = (src_x[i] - x_center) / x_scale;
            double v = (src_y[i] - y_center) / y_scale;
            double px = cx[0];
            double py = cy[0];
            int idx = 1;
            for (int di = 0; di <= sip_order; di++) {
                for (int dj = 0; dj <= sip_order - di; dj++) {
                    if (di == 0 && dj == 0) continue;
                    double term = pow(u, di) * pow(v, dj);
                    px += cx[idx] * term;
                    py += cy[idx] * term;
                    idx++;
                }
            }
            double rx = dst_x[i] - px;
            double ry = dst_y[i] - py;
            sum_rms += rx * rx + ry * ry;
        }
        *out_rms = sqrt(sum_rms / n_img);
        trans.rms = *out_rms;
        trans.match_count = n_img;
        
        SIP_LOG("  RMS after fit: %.4f px", *out_rms);
        
        if (out_trans) {
            trans.x00 = cx[0];
            trans.x10 = cx[1] / x_scale;
            trans.x01 = cx[2] / y_scale;
            trans.y00 = cy[0];
            trans.y10 = cy[1] / x_scale;
            trans.y01 = cy[2] / y_scale;
            
            if (sip_order >= 2) {
                trans.x20 = cx[3] / (x_scale * x_scale);
                trans.x11 = cx[4] / (x_scale * y_scale);
                trans.x02 = cx[5] / (y_scale * y_scale);
                trans.y20 = cy[3] / (x_scale * x_scale);
                trans.y11 = cy[4] / (x_scale * y_scale);
                trans.y02 = cy[5] / (y_scale * y_scale);
            }
            if (sip_order >= 3) {
                trans.x30 = cx[6] / (x_scale * x_scale * x_scale);
                trans.x21 = cx[7] / (x_scale * x_scale * y_scale);
                trans.x12 = cx[8] / (x_scale * y_scale * y_scale);
                trans.x03 = cx[9] / (y_scale * y_scale * y_scale);
                trans.y30 = cy[6] / (x_scale * x_scale * x_scale);
                trans.y21 = cy[7] / (x_scale * x_scale * y_scale);
                trans.y12 = cy[8] / (x_scale * y_scale * y_scale);
                trans.y03 = cy[9] / (y_scale * y_scale * y_scale);
            }
            if (sip_order >= 4) {
                double x4 = x_scale * x_scale * x_scale * x_scale;
                double y4 = y_scale * y_scale * y_scale * y_scale;
                trans.x40 = cx[10] / x4; trans.x31 = cx[11] / (x_scale * x_scale * x_scale * y_scale);
                trans.x22 = cx[12] / (x_scale * x_scale * y_scale * y_scale);
                trans.x13 = cx[13] / (x_scale * y_scale * y_scale * y_scale); trans.x04 = cx[14] / y4;
                trans.y40 = cy[10] / x4; trans.y31 = cy[11] / (x_scale * x_scale * x_scale * y_scale);
                trans.y22 = cy[12] / (x_scale * x_scale * y_scale * y_scale);
                trans.y13 = cy[13] / (x_scale * y_scale * y_scale * y_scale); trans.y04 = cy[14] / y4;
            }
            if (sip_order >= 5) {
                double x5 = x_scale * x_scale * x_scale * x_scale * x_scale;
                double y5 = y_scale * y_scale * y_scale * y_scale * y_scale;
                trans.x50 = cx[15] / x5; trans.x41 = cx[16] / (x_scale * x_scale * x_scale * x_scale * y_scale);
                trans.x32 = cx[17] / (x_scale * x_scale * x_scale * y_scale * y_scale);
                trans.x23 = cx[18] / (x_scale * x_scale * y_scale * y_scale * y_scale);
                trans.x14 = cx[19] / (x_scale * y_scale * y_scale * y_scale * y_scale); trans.x05 = cx[20] / y5;
                trans.y50 = cy[15] / y5; trans.y41 = cy[16] / (x_scale * x_scale * x_scale * x_scale * y_scale);
                trans.y32 = cy[17] / (x_scale * x_scale * x_scale * y_scale * y_scale);
                trans.y23 = cy[18] / (x_scale * x_scale * y_scale * y_scale * y_scale);
                trans.y14 = cy[19] / (x_scale * y_scale * y_scale * y_scale * y_scale); trans.y05 = cy[20] / y5;
            }
            trans.order = sip_order;
            sip_trans_copy(out_trans, &trans);
        }
        
        SIP_LOG("  Full polynomial fit (order=%d) completed", sip_order);
    } else {
        double sum_rms = 0;
        for (int i = 0; i < n_img; i++) {
            double px, py;
            sip_trans_apply(&trans, src_x[i], src_y[i], &px, &py);
            double rx = dst_x[i] - px;
            double ry = dst_y[i] - py;
            sum_rms += rx * rx + ry * ry;
        }
        *out_rms = sqrt(sum_rms / n_img);
        trans.rms = *out_rms;
        trans.match_count = n_img;
        SIP_LOG("  RMS after fit: %.4f px", *out_rms);
        if (out_trans) sip_trans_copy(out_trans, &trans);
    }
    
    if (out_sip && sip_order > 1) {
        int status = sip_coeffs_compute(&trans, img_w, img_h, out_sip);
        if (status) {
            SIP_LOG("  SIP coefficient calculation failed");
            out_sip->valid = 0;
        } else {
            SIP_LOG("  SIP coefficients computed successfully");
        }
    }
    
    SIP_LOG("=== SIP Fitting Complete ===");
    return 0;
}
