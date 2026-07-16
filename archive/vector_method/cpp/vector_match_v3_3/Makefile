CXX = g++
CXXFLAGS = -O3 -march=native -Wall -std=c++17 -fopenmp
LDFLAGS = -static-libgcc -static-libstdc++ -shared -lm -fopenmp

SRCDIR = src
INCDIR = include
TARGET = vector_match_v3_3.dll

# 复用V2的third_party
THIRD_PARTY = ../vector_match_v2/third_party

SOURCES = $(SRCDIR)/vm33_core.cpp

all: $(TARGET)

$(TARGET): $(SOURCES)
	$(CXX) $(CXXFLAGS) $(LDFLAGS) -o $@ $^ -I$(INCDIR) -I$(THIRD_PARTY)

clean:
	del /f /q $(TARGET) 2>nul
	del /f /q $(SRCDIR)\*.o 2>nul

.PHONY: all clean
