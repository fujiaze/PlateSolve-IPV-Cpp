# ============================================================================
# Makefile - V4.2 PairVerifier 模块（Task 5）
#
# 编译为 pair_verifier.dll，C++17，依赖 Eigen3（SVD）
# 无 OpenMP（单线程验证）
# ============================================================================

CXX = g++
CXXFLAGS = -O2 -march=native -Wall -std=c++17 -I../common -Iinclude -I../../vector_match_v2/third_party
LDFLAGS = -static -lm

DLL = pair_verifier.dll
SRCS = src/pv_core.cpp src/pv_mad.cpp src/pv_bayes.cpp src/pv_triangle.cpp

TEST_TARGET = test/test_pv.exe

all: $(DLL)

# 编译 DLL
$(DLL): $(SRCS) include/pv_api.h src/pv_internal.h
	$(CXX) $(CXXFLAGS) -shared -o $@ $(SRCS) $(LDFLAGS)
	@echo "DLL 编译成功: $(DLL)"

# 编译并运行单元测试
test_pv: $(TEST_TARGET)
	$(TEST_TARGET)

$(TEST_TARGET): test/test_pv.cpp $(SRCS) include/pv_api.h src/pv_internal.h
	@if not exist test mkdir test
	$(CXX) $(CXXFLAGS) -o $@ test/test_pv.cpp $(SRCS) $(LDFLAGS)

# 仅检查语法
check: all
	@echo "Build OK"

clean:
	del /f /q $(DLL) 2>nul
	del /f /q $(TEST_TARGET) 2>nul

.PHONY: all check clean test_pv
