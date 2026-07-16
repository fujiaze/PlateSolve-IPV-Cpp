$ErrorActionPreference = "Continue"
Set-Location "f:\Astro dev\Astro CS Normalization Database\lib\plate_solve"

$gxx = "C:\msys64\mingw64\bin\g++.exe"
$args = @(
    "-O2", "-march=native", "-Wall", "-std=c++17", "-fopenmp",
    "-Iinclude", "-Isrc", "-I..\gaia_xpsd_client\src", "-I..\dynamic_psf\include",
    "-Imodules\iterative_refine", "-Imodules\star_alignment",
    "-I..\star_detector\include", "-I..\star_detector\src",
    "-I..\astro_image_io\include", "-I..\astro_image_io\src",
    "-shared", "-o", "plate_solve.dll",
    "src\psolve_api.cpp", "src\psolve_log.cpp", "src\psolve_fov.cpp",
    "src\psolve_projection.cpp", "src\psolve_triangle.cpp", "src\psolve_ransac.cpp",
    "src\psolve_coarse.cpp", "src\psolve_fine.cpp",
    "modules\iterative_refine\psm_iterative_refine.cpp",
    "modules\iterative_refine\psm_sip.cpp",
    "modules\iterative_refine\psm_grid_match.cpp",
    "modules\star_alignment\psm_star_alignment.cpp",
    "..\star_detector\src\sdet_api.cpp",
    "..\star_detector\src\sdet_detector.cpp",
    "..\star_detector\src\sdet_image.cpp",
    "..\star_detector\src\sdet_log.cpp",
    "..\star_detector\src\sdet_background.cpp",
    "..\astro_image_io\src\aio_api.cpp",
    "..\astro_image_io\src\aio_fits.cpp",
    "..\astro_image_io\src\aio_xisf.cpp",
    "..\astro_image_io\src\aio_log.cpp",
    "..\gaia_xpsd_client\src\gaia_client.c",
    "-static-libgcc", "-static-libstdc++", "-lm", "-lz", "-lgomp"
)

Write-Host "Building plate_solve.dll..."
$process = Start-Process -FilePath $gxx -ArgumentList $args -NoNewWindow -Wait -PassThru -RedirectStandardOutput "build_out.txt" -RedirectStandardError "build_err.txt"

Write-Host "Exit code: $($process.ExitCode)"

if (Test-Path "build_err.txt") {
    $err = Get-Content "build_err.txt" -Raw
    if ($err.Length > 0) {
        Write-Host "=== Build errors ==="
        Write-Host $err
    }
}

if (Test-Path "plate_solve.dll") {
    Write-Host "Build successful!"
    Get-Item "plate_solve.dll" | Select-Object Name, Length, LastWriteTime
} else {
    Write-Host "Build failed - DLL not created"
}