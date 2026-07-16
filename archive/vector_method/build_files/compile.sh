#!/bin/bash
cd "/f/Astro dev/Astro CS Normalization Database/lib/plate_solve"

echo "Compiling plate_solve.dll..."

g++ -O2 -march=native -Wall -std=c++17 -fopenmp \
    -Iinclude -Isrc \
    -I../gaia_xpsd_client/src \
    -I../dynamic_psf/include \
    -Imodules/iterative_refine \
    -Imodules/star_alignment \
    -I../star_detector/include \
    -I../star_detector/src \
    -I../astro_image_io/include \
    -I../astro_image_io/src \
    -shared -o plate_solve.dll \
    src/psolve_api.cpp \
    src/psolve_log.cpp \
    src/psolve_fov.cpp \
    src/psolve_projection.cpp \
    src/psolve_triangle.cpp \
    src/psolve_ransac.cpp \
    src/psolve_coarse.cpp \
    src/psolve_fine.cpp \
    modules/iterative_refine/psm_iterative_refine.cpp \
    modules/iterative_refine/psm_sip.cpp \
    modules/iterative_refine/psm_grid_match.cpp \
    modules/star_alignment/psm_star_alignment.cpp \
    ../star_detector/src/sdet_api.cpp \
    ../star_detector/src/sdet_detector.cpp \
    ../star_detector/src/sdet_image.cpp \
    ../star_detector/src/sdet_log.cpp \
    ../star_detector/src/sdet_background.cpp \
    ../astro_image_io/src/aio_api.cpp \
    ../astro_image_io/src/aio_fits.cpp \
    ../astro_image_io/src/aio_xisf.cpp \
    ../astro_image_io/src/aio_log.cpp \
    ../gaia_xpsd_client/src/gaia_client.c \
    -static-libgcc -static-libstdc++ -lm -lz -lgomp

if [ -f plate_solve.dll ]; then
    echo "Build successful!"
    ls -la plate_solve.dll
else
    echo "Build failed!"
fi