@echo off
setlocal EnableDelayedExpansion

set SCRIPT_DIR=%~dp0
set MODULE_DIR=%SCRIPT_DIR%
set INCLUDE_DIR=%MODULE_DIR%..\..\include
set COMMON_DIR=%MODULE_DIR%..\common
set OUTPUT_DIR=%MODULE_DIR%

set CXX=cl
set CXXFLAGS=/O2 /EHsc /DNDEBUG /D_CRT_SECURE_NO_WARNINGS
set LDFLAGS=/DLL

set SRC=%MODULE_DIR%psm_iterative_refine.cpp
set OBJ=%OUTPUT_DIR%psm_iterative_refine.obj
set DLL=%OUTPUT_DIR%psm_iterative_refine.dll
set LIB=%OUTPUT_DIR%psm_iterative_refine.lib
set PDB=%OUTPUT_DIR%psm_iterative_refine.pdb

echo [BUILD] Compiling psm_iterative_refine.cpp...
%CXX% %CXXFLAGS% /I"%INCLUDE_DIR%" /I"%COMMON_DIR%" /c "%SRC%" /Fo:"%OBJ%"
if errorlevel 1 (
    echo [ERROR] Compilation failed
    exit /b 1
)

echo [BUILD] Linking psm_iterative_refine.dll...
link %LDFLAGS% "%OBJ%" /OUT:"%DLL%" /IMPLIB:"%LIB%" /PDB:"%PDB%"
if errorlevel 1 (
    echo [ERROR] Linking failed
    exit /b 1
)

echo [BUILD] Cleaning up...
del "%OBJ%" 2>nul

echo [BUILD] Build successful: %DLL%
echo [BUILD] Library: %LIB%

endlocal
