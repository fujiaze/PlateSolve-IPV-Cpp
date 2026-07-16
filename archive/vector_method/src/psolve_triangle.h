#ifndef PSOLVE_TRIANGLE_H
#define PSOLVE_TRIANGLE_H

typedef struct {
    int a_idx, b_idx, c_idx;
    double ba_ratio;
    double ca_ratio;
    double side_a_angle;
    double side_a_length;
} PSolveTriangle;

typedef struct {
    int img_idx;
    int cat_idx;
} PSolveStarPair;

int psolve_build_triangles(const double *x, const double *y, int n,
                            int nbright,
                            PSolveTriangle **out_tris, int *out_count);
int psolve_match_triangles(
    const PSolveTriangle *tris_a, int na,
    const PSolveTriangle *tris_b, int nb,
    double radius, double min_scale, double max_scale,
    PSolveStarPair **out_pairs, int *out_pair_count);
void psolve_free_triangles(PSolveTriangle *tris);
void psolve_free_pairs(PSolveStarPair *pairs);

#endif
