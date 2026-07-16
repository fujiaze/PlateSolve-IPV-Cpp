# ============================================================================
# Makefile - V4.3 单一 DLL 构建脚本
#
# 将 V4.2 的 5 个独立 DLL 合并为单一 vector_match_v4_3.dll
# 编译所有 src/*.cpp 为单一 DLL, C++ 内部直接函数调用, 无 ctypes 边界
#
# 依赖: Eigen3 (头文件), OpenMP
# Eigen 路径: ../../vector_match_v2/third_party (与 V4.2 共享)
# ============================================================================

CXX = g++
CXXFLAGS = -O2 -march=native -Wall -std=c++17 -fopenmp \
           -Iinclude -I../vector_match_v2/third_party
LDFLAGS = -static-libgcc -static-libstdc++ -fopenmp

# 单一 DLL 目标
DLL = vector_match_v4_3.dll

# 所有源文件 (5 模块 + IRM + 入口)
SRCS = \
    src/vm43_select.cpp \
    src/vm43_match.cpp \
    src/vm43_expand.cpp \
    src/vm43_geometry.cpp \
    src/vm43_verify.cpp \
    src/vm43_fit.cpp \
    src/vm43_score.cpp \
    src/vm43_irm.cpp \
    src/vm43_entry.cpp

# 头文件依赖
HEADERS = \
    include/vm43_types.h \
    include/vm43_api.h \
    include/vm43_internal.h \
    include/vm43_log.h

# ============================================================================
# 主要目标
# ============================================================================

all: $(DLL)

$(DLL): $(SRCS) $(HEADERS)
	$(CXX) $(CXXFLAGS) -shared -o $@ $(SRCS) $(LDFLAGS)
	@echo "=== V4.3 单一 DLL 编译成功: $(DLL) ==="

# 仅编译验证(不链接), 检查语法
check: $(SRCS)
	@for f in $(SRCS); do \
		echo "Compiling $$f..."; \
		$(CXX) $(CXXFLAGS) -c -o /dev/null $$f || exit 1; \
	done
	@echo "=== 所有源文件编译通过 ==="

# ============================================================================
# 单元测试目标
# ============================================================================

# 编译并运行所有单元测试
test: test_select test_match test_expand test_geometry test_verify test_score test_fit test_irm test_solve
	@echo "=== 所有单元测试通过 ==="

# Task 2: StarSelector + VectorMatcher 测试
# vm43_entry.cpp 提供 get_gaia_client_handle()/get_star_detector_handle() 定义
test_select: test/test_select.cpp src/vm43_select.cpp src/vm43_entry.cpp $(HEADERS)
	@if not exist test mkdir test
	$(CXX) $(CXXFLAGS) -o test/test_select.exe test/test_select.cpp src/vm43_select.cpp src/vm43_entry.cpp $(LDFLAGS)
	test/test_select.exe

test_match: test/test_match.cpp src/vm43_match.cpp $(HEADERS)
	@if not exist test mkdir test
	$(CXX) $(CXXFLAGS) -o test/test_match.exe test/test_match.cpp src/vm43_match.cpp $(LDFLAGS)
	test/test_match.exe

# Task 3: PairExpander 测试
test_expand: test/test_expand.cpp src/vm43_expand.cpp $(HEADERS)
	@if not exist test mkdir test
	$(CXX) $(CXXFLAGS) -o test/test_expand.exe test/test_expand.cpp src/vm43_expand.cpp $(LDFLAGS)
	test/test_expand.exe

# Task 4: 局部几何一致性过滤测试
test_geometry: test/test_geometry.cpp src/vm43_geometry.cpp $(HEADERS)
	@if not exist test mkdir test
	$(CXX) $(CXXFLAGS) -o test/test_geometry.exe test/test_geometry.cpp src/vm43_geometry.cpp $(LDFLAGS)
	test/test_geometry.exe

# Task 5: PairVerifier 测试
test_verify: test/test_verify.cpp src/vm43_verify.cpp $(HEADERS)
	@if not exist test mkdir test
	$(CXX) $(CXXFLAGS) -o test/test_verify.exe test/test_verify.cpp src/vm43_verify.cpp $(LDFLAGS)
	test/test_verify.exe

# Task 6: S_robust 评分测试
test_score: test/test_score.cpp src/vm43_score.cpp $(HEADERS)
	@if not exist test mkdir test
	$(CXX) $(CXXFLAGS) -o test/test_score.exe test/test_score.cpp src/vm43_score.cpp $(LDFLAGS)
	test/test_score.exe

# Task 7: WcsFitter 测试
test_fit: test/test_fit.cpp src/vm43_fit.cpp $(HEADERS)
	@if not exist test mkdir test
	$(CXX) $(CXXFLAGS) -o test/test_fit.exe test/test_fit.cpp src/vm43_fit.cpp $(LDFLAGS)
	test/test_fit.exe

# Task 8: IRM 闭环测试
test_irm: test/test_irm.cpp src/vm43_irm.cpp src/vm43_expand.cpp src/vm43_geometry.cpp src/vm43_verify.cpp src/vm43_fit.cpp src/vm43_score.cpp $(HEADERS)
	@if not exist test mkdir test
	$(CXX) $(CXXFLAGS) -o test/test_irm.exe test/test_irm.cpp \
		src/vm43_irm.cpp src/vm43_expand.cpp src/vm43_geometry.cpp \
		src/vm43_verify.cpp src/vm43_fit.cpp src/vm43_score.cpp $(LDFLAGS)
	test/test_irm.exe

# Task 9: 入口测试
test_solve: test/test_solve.cpp $(SRCS) $(HEADERS)
	@if not exist test mkdir test
	$(CXX) $(CXXFLAGS) -o test/test_solve.exe test/test_solve.cpp $(SRCS) $(LDFLAGS)
	test/test_solve.exe

# Debug: 分步调试测试 (合成数据验证 3 类失败场景)
test_debug: test/test_debug.cpp src/vm43_irm.cpp src/vm43_expand.cpp src/vm43_geometry.cpp src/vm43_verify.cpp src/vm43_fit.cpp src/vm43_score.cpp src/vm43_match.cpp $(HEADERS)
	@if not exist test mkdir test
	$(CXX) $(CXXFLAGS) -o test/test_debug.exe test/test_debug.cpp \
		src/vm43_irm.cpp src/vm43_expand.cpp src/vm43_geometry.cpp \
		src/vm43_verify.cpp src/vm43_fit.cpp src/vm43_score.cpp src/vm43_match.cpp $(LDFLAGS)
	test/test_debug.exe

# ============================================================================
# 清理
# ============================================================================

clean:
	del /f /q $(DLL) 2>nul
	del /f /q test\*.exe 2>nul
	del /f /q src\*.o 2>nul
	@echo "=== 清理完成 ==="

.PHONY: all check clean test test_select test_match test_expand test_geometry test_verify test_score test_fit test_irm test_solve
