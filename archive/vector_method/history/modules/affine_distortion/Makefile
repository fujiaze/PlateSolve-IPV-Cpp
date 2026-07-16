CXX = g++
CXXFLAGS = -O2 -march=native -Wall -std=c++17
LDFLAGS = -static-libgcc -static-libstdc++ -shared -lm
INCDIR = -I../common
TARGET = affine_distortion.dll
SOURCES = psm_affine_distortion.cpp

all: $(TARGET)

$(TARGET): $(SOURCES)
	$(CXX) $(CXXFLAGS) $(LDFLAGS) $(INCDIR) -o $@ $^

clean:
	del /f /q $(TARGET) 2>nul

.PHONY: all clean
