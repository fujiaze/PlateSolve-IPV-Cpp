CXX = g++
CXXFLAGS = -O2 -march=native -Wall -std=c++17
LDFLAGS = -static-libgcc -static-libstdc++ -shared -fopenmp
INCDIR = -I../common -I../../../gaia_xpsd_client/src
LIBDIR = -L"C:\msys64\mingw64\lib"
LIBS = -lm -lz
TARGET = rms_calc.dll
SOURCES = psm_rms_calc.cpp
GAIA_SRC = ../../../gaia_xpsd_client/src/gaia_client.c

all: $(TARGET)

$(TARGET): $(SOURCES) $(GAIA_SRC)
	$(CXX) $(CXXFLAGS) $(LDFLAGS) $(INCDIR) $(LIBDIR) -o $@ $(SOURCES) $(GAIA_SRC) $(LIBS)

clean:
	del /f /q $(TARGET) 2>nul

.PHONY: all clean
