@echo off
cd /d "f:\Astro dev\Astro CS Normalization Database\lib\plate_solve\modules\star_alignment"
C:\msys64\mingw64\bin\g++.exe -O2 -march=native -Wall -std=c++17 -shared -o star_alignment.dll psm_star_alignment.cpp -static-libgcc -static-libstdc++
echo Exit code: %ERRORLEVEL%
