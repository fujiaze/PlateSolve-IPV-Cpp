#ifndef PSOLVE_FINE_H
#define PSOLVE_FINE_H

#include "plate_solve.h"

typedef struct PSolveHandle_s PSolveHandleInternal;

int psolve_fine_solve(
    PSolveHandleInternal *handle,
    const uint16_t *image, int width, int height,
    const PSolveCoarseResult *coarse,
    const PSolveImageData *img_data,
    PSolveFineResult *result);

#endif
