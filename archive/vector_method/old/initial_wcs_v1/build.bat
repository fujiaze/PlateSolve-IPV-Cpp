@echo off
cd /d "f:\Astro dev\Astro CS Normalization Database\lib\plate_solve\modules\initial_wcs"

echo === 编译 psm_initial_wcs.dll ===

C:\msys64\mingw64\bin\g++.exe -O2 -march=native -Wall -std=c++17 -shared ^
  -o psm_initial_wcs.dll ^
  psm_initial_wcs.cpp ^
  ..\..\src\psolve_projection.cpp ^
  ..\..\..\gaia_xpsd_client\src\gaia_client.c ^
  -I..\..\src ^
  -I..\..\gaia_xpsd_client\src ^
  -lz ^
  -static-libgcc -static-libstdc++

echo Exit code: %ERRORLEVEL%
if %ERRORLEVEL% EQU 0 (
    echo 编译成功: psm_initial_wcs.dll
) else (
    echo 编译失败
)
