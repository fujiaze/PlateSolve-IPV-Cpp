CXX = g++
CXXFLAGS = -O3 -march=native -Wall -std=c++17 -fopenmp
LDFLAGS = -static-libgcc -static-libstdc++ -shared -lm -fopenmp

SRCDIR = src
INCDIR = include
TARGET = vector_match_v2.dll

SOURCES = $(SRCDIR)/vm2_core.cpp

all: $(TARGET)

$(TARGET): $(SOURCES)
	$(CXX) $(CXXFLAGS) $(LDFLAGS) -o $@ $^ -I$(INCDIR) -Ithird_party

clean:
	del /f /q $(TARGET) 2>nul
	del /f /q $(SRCDIR)\*.o 2>nul

.PHONY: all clean
