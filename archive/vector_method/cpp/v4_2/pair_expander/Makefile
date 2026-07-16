# PairExpander V4.2 Makefile
# 核心简化: 无 nanoflann / k-vector / Eigen / OpenMP 依赖, 纯线性扫描 NN
CXX = g++
CXXFLAGS = -O2 -march=native -Wall -std=c++17 -I../common -Iinclude
LDFLAGS = -static-libgcc -static-libstdc++ -static -lpthread

DLL = pair_expander.dll
SRCS = src/pe_core.cpp
TEST_SRC = test/test_pe.cpp

all: $(DLL)

$(DLL): $(SRCS)
	$(CXX) $(CXXFLAGS) -shared -o $@ $^ $(LDFLAGS)

check: all
	@echo "Build OK"

# 编译测试程序 (直接编译源文件 + 测试, 不依赖 DLL 运行时路径)
test_pe: $(TEST_SRC) $(SRCS)
	$(CXX) $(CXXFLAGS) -o test_pe.exe $(TEST_SRC) $(SRCS) $(LDFLAGS)
	@echo "Test built: test_pe.exe"

# 编译并运行测试
test: test_pe
	./test_pe.exe

clean:
	del /Q $(DLL) test_pe.exe 2>nul || rm -f $(DLL) test_pe.exe

.PHONY: all check test_pe test clean
