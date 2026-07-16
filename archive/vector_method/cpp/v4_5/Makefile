# ============================================================================
# Makefile - V4.5 单一 DLL 构建脚本
#
# V4.5 严格按 v4_4_relvec_sampling_design.md 实现 Phase A (相对向量法 θ 求解):
#   - Phase 0 (StarSelector): 复用 V4.4 算法  -> vm45_select.cpp
#   - Phase A (相对向量法):   绝对距离 k-vector + 1D θ 直方图 + 高斯平滑
#                            -> vm45_relvec.cpp
#   - Phase B (Entry/API):   DLL 入口与导出封装 -> vm45_entry.cpp
#   - 跳过 Phase B / IRM / WcsFitter
#
# 编译 src/*.cpp 为单一 vector_match_v4_5.dll, 与 vector_match_v4_4.dll 并存
#
# 依赖: Eigen3 (头文件, 可选), OpenMP
# Eigen 路径: ../vector_match_v2/third_party (与 V4.4 共享)
# ============================================================================

SHELL = C:/msys64/usr/bin/sh.exe

CXX = g++
CXXFLAGS = -O3 -march=native -funroll-loops -ffast-math -Wall -std=c++17 -fopenmp \
           -Iinclude -I../v4_4/include -I../vector_match_v2/third_party \
           -DVM45_EXPORTS
LDFLAGS = -shared -static-libgcc -static-libstdc++ -fopenmp

# 单一 DLL 目标
TARGET = vector_match_v4_5.dll

# 源文件 (V4.5 共 3 个源文件)
SRCS = \
    src/vm45_select.cpp \
    src/vm45_relvec.cpp \
    src/vm45_entry.cpp

OBJS = $(SRCS:.cpp=.o)

# 头文件依赖
HEADERS = \
    include/vm45_types.h \
    include/vm45_api.h \
    include/vm45_internal.h

# ============================================================================
# 主要目标
# ============================================================================

all: $(TARGET)

# 链接 DLL
$(TARGET): $(OBJS)
	$(CXX) $(LDFLAGS) -o $@ $^
	@echo "=== V4.5 单一 DLL 编译成功: $(TARGET) ==="

# 编译规则
%.o: %.cpp
	$(CXX) $(CXXFLAGS) -c -o $@ $<

# ============================================================================
# 语法检查 (仅编译不链接)
# ============================================================================

check:
	@for f in $(SRCS); do \
		printf "Checking %s...\n" "$$f"; \
		$(CXX) $(CXXFLAGS) -fsyntax-only $$f || exit 1; \
	done
	@printf "=== 所有源文件语法检查通过 ===\n"

# ============================================================================
# 清理
# ============================================================================

clean:
	rm -f $(OBJS) $(TARGET)
	@echo "=== 清理完成 ==="

.PHONY: all check clean
