CXX = g++
CXXFLAGS = -O3 -march=native -Wall -std=c++17 -fopenmp
LDFLAGS = -shared -static-libgcc -static-libstdc++ -lm -fopenmp
TARGET = vector_match_v3_4.dll
SRCDIR = src
INCDIR = include
THIRDPARTY = ../vector_match_v2/third_party

SRCS = $(SRCDIR)/vm34_core.cpp

all: $(TARGET)

$(TARGET): $(SRCS)
	$(CXX) $(CXXFLAGS) $(LDFLAGS) -o $@ $< -I$(INCDIR) -I$(THIRDPARTY)

clean:
	del /f /q $(TARGET) 2>nul
	del /f /q $(SRCDIR)\*.o 2>nul

.PHONY: all clean
