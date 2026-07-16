$ErrorActionPreference = "Continue"
Set-Location "f:\Astro dev\Astro CS Normalization Database\lib\plate_solve"

$gxx = "C:\msys64\mingw64\bin\g++.exe"

Write-Host "Testing g++ with verbose output..."
$process = Start-Process -FilePath $gxx -ArgumentList "-v", "-c", "src\psolve_log.cpp", "-Iinclude", "-Isrc", "-o", "test.obj" -NoNewWindow -Wait -PassThru -RedirectStandardError "verbose_err.txt"

Write-Host "Exit code: $($process.ExitCode)"
Write-Host ""
Write-Host "=== Verbose error output ==="
Get-Content "verbose_err.txt" -ErrorAction SilentlyContinue | Select-Object -Last 50

if (Test-Path "test.obj") {
    Write-Host "Compile OK"
    Remove-Item "test.obj"
} else {
    Write-Host "Compile FAILED"
}