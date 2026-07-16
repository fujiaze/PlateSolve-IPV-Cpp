# ============================================================================
# Makefile - V4.4 单一 DLL 构建脚本
#
# V4.3 基础上新增 vm44_relvec.cpp (相对向量法完全替代单θ Phase A)
# 编译所有 src/*.cpp 为单一 vector_match_v4_4.dll
#
# 依赖: Eigen3 (头文件), OpenMP
# Eigen 路径: ../../vector_match_v2/third_party (与 V4.3 共享)
# ============================================================================

CXX = g++
CXXFLAGS = -O3 -march=native -funroll-loops -ffast-math -Wall -std=c++17 -fopenmp \
           -Iinclude -I../vector_match_v2/third_party
LDFLAGS = -static-libgcc -static-libstdc++ -fopenmp

# Windows MSYS2 环境用 sh 执行 recipe
SHELL = C:/msys64/usr/bin/sh.exe

# 单一 DLL 目标
DLL = vector_match_v4_4.dll

# 所有源文件 (V4.3 9模块 + V4.4 新增 vm44_relvec.cpp)
SRCS = \
    src/vm44_select.cpp \
    src/vm44_match.cpp \
    src/vm44_relvec.cpp \
    src/vm44_expand.cpp \
    src/vm44_geometry.cpp \
    src/vm44_verify.cpp \
    src/vm44_fit.cpp \
    src/vm44_score.cpp \
    src/vm44_irm.cpp \
    src/vm44_entry.cpp

# 头文件依赖
HEADERS = \
    include/vm44_types.h \
    include/vm44_api.h \
    include/vm44_internal.h \
    include/vm44_log.h

# ============================================================================
# 主要目标
# ============================================================================

all: $(DLL)

$(DLL): $(SRCS) $(HEADERS)
	$(CXX) $(CXXFLAGS) -shared -o $@ $(SRCS) $(LDFLAGS)
	@echo "=== V4.4 单一 DLL 编译成功: $(DLL) ==="

# 仅编译验证(不链接), 检查语法
check: $(SRCS)
	@for f in $(SRCS); do \
		printf "Compiling %s...\n" "$$f"; \
		$(CXX) $(CXXFLAGS) -c -o /dev/null $$f || exit 1; \
	done
	@printf "=== 所有源文件编译通过 ===\n"

# ============================================================================
# 单元测试目标
# ============================================================================

test: test_select test_match test_relvec test_expand test_geometry test_verify test_score test_fit test_irm test_solve
	@echo "=== 所有单元测试通过 ==="

test_select: test/test_vm44_select.cpp src/vm44_select.cpp src/vm44_entry.cpp $(HEADERS)
	@if not exist test mkdir test
	$(CXX) $(CXXFLAGS) -o test/test_vm44_select.exe test/test_vm44_select.cpp src/vm44_select.cpp src/vm44_entry.cpp $(LDFLAGS)
	test/test_vm44_select.exe

test_match: test/test_vm44_match.cpp src/vm44_match.cpp src/vm44_relvec.cpp $(HEADERS)
	@if not exist test mkdir test
	$(CXX) $(CXXFLAGS) -o test/test_vm44_match.exe test/test_vm44_match.cpp src/vm44_match.cpp src/vm44_relvec.cpp $(LDFLAGS)
	test/test_vm44_match.exe

# V4.4 新增: 相对向量法单元测试
test_relvec: test/test_vm44_relvec.cpp src/vm44_relvec.cpp $(HEADERS)
	@if not exist test mkdir test
	$(CXX) $(CXXFLAGS) -o test/test_vm44_relvec.exe test/test_vm44_relvec.cpp src/vm44_relvec.cpp $(LDFLAGS)
	test/test_vm44_relvec.exe

test_expand: test/test_vm44_expand.cpp src/vm44_expand.cpp $(HEADERS)
	@if not exist test mkdir test
	$(CXX) $(CXXFLAGS) -o test/test_vm44_expand.exe test/test_vm44_expand.cpp src/vm44_expand.cpp $(LDFLAGS)
	test/test_vm44_expand.exe

test_geometry: test/test_vm44_geometry.cpp src/vm44_geometry.cpp $(HEADERS)
	@if not exist test mkdir test
	$(CXX) $(CXXFLAGS) -o test/test_vm44_geometry.exe test/test_vm44_geometry.cpp src/vm44_geometry.cpp $(LDFLAGS)
	test/test_vm44_geometry.exe

test_verify: test/test_vm44_verify.cpp src/vm44_verify.cpp $(HEADERS)
	@if not exist test mkdir test
	$(CXX) $(CXXFLAGS) -o test/test_vm44_verify.exe test/test_vm44_verify.cpp src/vm44_verify.cpp $(LDFLAGS)
	test/test_vm44_verify.exe

test_score: test/test_vm44_score.cpp src/vm44_score.cpp $(HEADERS)
	@if not exist test mkdir test
	$(CXX) $(CXXFLAGS) -o test/test_vm44_score.exe test/test_vm44_score.cpp src/vm44_score.cpp $(LDFLAGS)
	test/test_vm44_score.exe

test_fit: test/test_vm44_fit.cpp src/vm44_fit.cpp $(HEADERS)
	@if not exist test mkdir test
	$(CXX) $(CXXFLAGS) -o test/test_vm44_fit.exe test/test_vm44_fit.cpp src/vm44_fit.cpp $(LDFLAGS)
	test/test_vm44_fit.exe

test_irm: test/test_vm44_irm.cpp src/vm44_irm.cpp src/vm44_expand.cpp src/vm44_geometry.cpp src/vm44_verify.cpp src/vm44_fit.cpp src/vm44_score.cpp $(HEADERS)
	@if not exist test mkdir test
	$(CXX) $(CXXFLAGS) -o test/test_vm44_irm.exe test/test_vm44_irm.cpp \
		src/vm44_irm.cpp src/vm44_expand.cpp src/vm44_geometry.cpp \
		src/vm44_verify.cpp src/vm44_fit.cpp src/vm44_score.cpp $(LDFLAGS)
	test/test_vm44_irm.exe

test_solve: test/test_vm44_solve.cpp $(SRCS) $(HEADERS)
	@if not exist test mkdir test
	$(CXX) $(CXXFLAGS) -o test/test_vm44_solve.exe test/test_vm44_solve.cpp $(SRCS) $(LDFLAGS)
	test/test_vm44_solve.exe

# ============================================================================
# 清理
# ============================================================================

clean:
	del /f /q $(DLL) 2>nul
	del /f /q test\*.exe 2>nul
	del /f /q src\*.o 2>nul
	@echo "=== 清理完成 ==="

.PHONY: all check clean test test_select test_match test_relvec test_expand test_geometry test_verify test_score test_fit test_irm test_solve
