GXX = C:\msys64\mingw64\bin\g++.exe
CFLAGS = -O2 -march=native -Wall -std=c++17

SRC = psm_iterative_refine.cpp
DLL = psm_iterative_refine.dll

all: $(DLL)

$(DLL): $(SRC)
	$(GXX) $(CFLAGS) -shared -o $(DLL) $(SRC) -static-libgcc -static-libstdc++

clean:
	del /f $(DLL) 2>nul

.PHONY: all clean
