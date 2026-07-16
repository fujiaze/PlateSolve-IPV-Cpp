#ifndef PSOLVE_COARSE_H
#define PSOLVE_COARSE_H

#include "plate_solve.h"

#ifdef __cplusplus
extern "C" {
#endif

int psolve_coarse_solve(
    void *handle,
    const uint16_t *image, int width, int height,
    const PSolveImageData *img_data,
    const double *det_x, const double *det_y, int det_count,
    PSolveCoarseResult *result);

#ifdef __cplusplus
}
#endif

#endif
