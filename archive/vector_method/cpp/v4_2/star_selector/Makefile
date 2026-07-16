# Makefile - V4.2 StarSelector 模块
# 编译为 star_selector.dll，C++17 单线程（密度匹配无需 OpenMP）

CXX = g++
CXXFLAGS = -O2 -march=native -Wall -std=c++17 -I../common -Iinclude
LDFLAGS = -static-libgcc -static-libstdc++

DLL = star_selector.dll
SRCS = src/ss_core.cpp

# 测试程序
TEST = test_ss.exe
TEST_SRC = test/test_ss.cpp

# MinGW 路径（Windows）
MINGW_BIN = C:\msys64\mingw64\bin

all: $(DLL)

$(DLL): $(SRCS)
	$(CXX) $(CXXFLAGS) -shared -o $@ $^ $(LDFLAGS)

check: all
	@echo "Build OK"

# 编译并运行单元测试
test: $(TEST)
	./$(TEST)

$(TEST): $(TEST_SRC) $(DLL)
	$(CXX) $(CXXFLAGS) -o $@ $(TEST_SRC) -L. -lstar_selector $(LDFLAGS)

clean:
	del /Q $(DLL) 2>nul || rm -f $(DLL)
	del /Q $(TEST) 2>nul || rm -f $(TEST)

.PHONY: all check test clean
