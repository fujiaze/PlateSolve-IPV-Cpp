#ifndef PSM_COMMON_H
#define PSM_COMMON_H

#include <stddef.h>
#include <stdint.h>

#ifdef _WIN32
#define PSM_EXPORT __declspec(dllexport)
#else
#define PSM_EXPORT __attribute__((visibility("default")))
#endif

#ifdef __cplusplus
extern "C" {
#endif

#define PSM_OK                0
#define PSM_ERR_NO_MATCH      1
#define PSM_ERR_NOT_ENOUGH    2
#define PSM_ERR_INVALID_PARAM 3
#define PSM_ERR_SINGULAR      4
#define PSM_ERR_ALLOC         5
#define PSM_ERR_NO_MEMORY     6
#define PSM_ERR_NO_DATA       7
#define PSM_ERR_MATH          8

typedef struct {
    double a0, a1, a2;
    double b0, b1, b2;
} PSMAffine;

typedef struct {
    double a0, a1, a2, a3, a4, a5;
    double b0, b1, b2, b3, b4, b5;
} PSMDistortion;

typedef struct {
    int img_idx;
    int cat_idx;
} PSMStarPair;

typedef struct {
    int a_idx, b_idx, c_idx;
    double ba_ratio;
    double ca_ratio;
    double side_a_angle;
    double side_a_length;
} PSMTriangle;

#ifdef __cplusplus
}
#endif

#endif
