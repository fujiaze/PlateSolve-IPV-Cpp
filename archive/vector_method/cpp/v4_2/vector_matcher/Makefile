# ============================================================================
# Makefile - V4.2 VectorMatcher 模块 (Task 3)
#
# 编译 vector_matcher.dll, 依赖 Eigen3 + OpenMP
#
# Eigen 路径: ../../vector_match_v2/third_party/Eigen (与 V4.1 共享)
# ============================================================================

CXX = g++
CXXFLAGS = -O2 -march=native -Wall -std=c++17 -fopenmp \
           -I../common -Iinclude \
           -I../../vector_match_v2/third_party
LDFLAGS = -static-libgcc -static-libstdc++ -fopenmp

DLL = vector_matcher.dll
SRCS = src/vm_core.cpp src/vm_prosac.cpp

# 测试目标
TEST_TARGET = test/test_vm.exe

all: $(DLL)

$(DLL): $(SRCS)
	$(CXX) $(CXXFLAGS) -shared -o $@ $^ $(LDFLAGS)

# 仅编译验证(不链接)
check: $(SRCS)
	$(CXX) $(CXXFLAGS) -c -o src/vm_core.o src/vm_core.cpp
	$(CXX) $(CXXFLAGS) -c -o src/vm_prosac.o src/vm_prosac.cpp
	@echo "Compile OK"

# 编译并运行单元测试 (静态编译所有源码, 避免依赖 DLL 路径)
test: $(TEST_TARGET)
	$(TEST_TARGET)

$(TEST_TARGET): test/test_vm.cpp $(SRCS) include/vm_api.h include/vm_prosac.h
	@if not exist test mkdir test
	$(CXX) $(CXXFLAGS) -o $@ test/test_vm.cpp $(SRCS) -Iinclude -I../common $(LDFLAGS)

clean:
	del /f /q $(DLL) 2>nul
	del /f /q $(TEST_TARGET) 2>nul
	del /f /q src\*.o 2>nul

.PHONY: all check clean test
