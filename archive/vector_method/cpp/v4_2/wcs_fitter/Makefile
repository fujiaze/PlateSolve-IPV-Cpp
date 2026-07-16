# ============================================================================
# Makefile - V4.2 WcsFitter 模块 (Task 6)
#
# 编译 wcs_fitter.dll, 依赖 Eigen3 (与 V4.1/V4.2 其他模块共享)
#
# Eigen 路径: ../../vector_match_v2/third_party (与 V4.1 共享)
# ============================================================================

CXX = g++
CXXFLAGS = -O2 -march=native -Wall -std=c++17 \
           -I../common -Iinclude \
           -I../../vector_match_v2/third_party
LDFLAGS = -static-libgcc -static-libstdc++

DLL = wcs_fitter.dll
SRCS = src/wf_core.cpp

# 测试目标
TEST_TARGET = test/test_wf.exe

all: $(DLL)

$(DLL): $(SRCS)
	$(CXX) $(CXXFLAGS) -shared -o $@ $^ $(LDFLAGS)

# 仅编译验证(不链接)
check: $(SRCS)
	$(CXX) $(CXXFLAGS) -c -o src/wf_core.o src/wf_core.cpp
	@echo "Compile OK"

# 编译并运行单元测试 (静态编译源码, 避免依赖 DLL 路径)
test: $(TEST_TARGET)
	$(TEST_TARGET)

$(TEST_TARGET): test/test_wf.cpp $(SRCS) include/wf_api.h
	@if not exist test mkdir test
	$(CXX) $(CXXFLAGS) -o $@ test/test_wf.cpp $(SRCS) $(LDFLAGS)

clean:
	del /f /q $(DLL) 2>nul
	del /f /q $(TEST_TARGET) 2>nul
	del /f /q src\*.o 2>nul

.PHONY: all check clean test
