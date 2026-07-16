$ErrorActionPreference = "Continue"
Set-Location "f:\Astro dev\Astro CS Normalization Database\lib\plate_solve"

Write-Host "Checking files..."
Get-ChildItem "src\*.cpp" | ForEach-Object { Write-Host $_.Name }
Get-ChildItem "modules\iterative_refine\*.cpp" | ForEach-Object { Write-Host $_.Name }

$gxx = "C:\msys64\mingw64\bin\g++.exe"
Write-Host "G++ path: $gxx"
Write-Host "G++ exists: $(Test-Path $gxx)"

Write-Host ""
Write-Host "Testing simple compile..."
& $gxx --version

Write-Host ""
Write-Host "Attempting single file compile test..."
$testResult = & $gxx -c "src\psolve_log.cpp" -Iinclude -Isrc -o "test.obj" 2>&1
Write-Host "Single file result: $testResult"
if (Test-Path "test.obj") {
    Write-Host "Single file compile OK"
    Remove-Item "test.obj"
} else {
    Write-Host "Single file compile FAILED"
}