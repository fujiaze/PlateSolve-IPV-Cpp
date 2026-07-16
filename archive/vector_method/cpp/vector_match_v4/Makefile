CXX = g++
CXXFLAGS = -O3 -march=native -Wall -Wextra -std=c++17 -fopenmp
LDFLAGS = -shared -static-libgcc -static-libstdc++ -lm -fopenmp
TARGET = vector_match_v4.dll
TEST_TARGET = test/test_kvector.exe
TEST_DENSITY = test/test_density.exe
TEST_BAYES = test/test_bayes.exe
TEST_BAYES_INC = test/test_bayes_increment.exe
SRCDIR = src
INCDIR = include
TESTDIR = test
THIRDPARTY = ../vector_match_v2/third_party

# DLL 源文件列表（vm4_density.cpp 由 Task 2 加入, vm4_prosac.cpp 由 Task 4 加入, vm4_bayes.cpp 由 Task 5 加入, vm4_triangle.cpp 由 Task 6 加入）
SRCS = $(SRCDIR)/vm4_core.cpp $(SRCDIR)/vm4_log.cpp $(SRCDIR)/vm4_kvector.cpp $(SRCDIR)/vm4_density.cpp $(SRCDIR)/vm4_prosac.cpp $(SRCDIR)/vm4_bayes.cpp $(SRCDIR)/vm4_triangle.cpp
PROSAC_TEST = test/test_prosac.exe
TEST_TRIANGLE = test/test_triangle.exe

all: $(TARGET)

$(TARGET): $(SRCS)
	$(CXX) $(CXXFLAGS) $(LDFLAGS) -o $@ $^ -I$(INCDIR) -I$(THIRDPARTY)

# 仅编译 vm4_kvector.cpp（验证无语法错误）
check_kvector: $(SRCDIR)/vm4_kvector.cpp $(INCDIR)/vm4_kvector.h
	$(CXX) $(CXXFLAGS) -c -o $(SRCDIR)/vm4_kvector.o $< -I$(INCDIR)
	@echo "vm4_kvector.cpp compiled OK"

# 编译并运行 k-vector 测试（独立程序，仅依赖 vm4_kvector.cpp）
test: $(TEST_TARGET)
	$(TEST_TARGET)

$(TEST_TARGET): $(TESTDIR)/test_kvector.cpp $(SRCDIR)/vm4_kvector.cpp $(INCDIR)/vm4_kvector.h
	@if not exist $(TESTDIR) mkdir $(TESTDIR)
	$(CXX) $(CXXFLAGS) -o $@ $(TESTDIR)/test_kvector.cpp $(SRCDIR)/vm4_kvector.cpp -I$(INCDIR) -lm

# Task 2: 编译并运行 vm4_density 单元测试
test_density: $(TEST_DENSITY)
	$(TEST_DENSITY)

$(TEST_DENSITY): $(TESTDIR)/test_density.cpp $(SRCDIR)/vm4_density.cpp $(INCDIR)/vm4_density.h
	@if not exist $(TESTDIR) mkdir $(TESTDIR)
	$(CXX) -O2 -std=c++17 -Wall -I$(INCDIR) -o $@ $(TESTDIR)/test_density.cpp $(SRCDIR)/vm4_density.cpp -lm

# Task 4: 仅编译 vm4_prosac.cpp（验证无语法错误）
check_prosac: $(SRCDIR)/vm4_prosac.cpp $(INCDIR)/vm4_prosac.h
	$(CXX) $(CXXFLAGS) -c -o $(SRCDIR)/vm4_prosac.o $< -I$(INCDIR)
	@echo "vm4_prosac.cpp compiled OK"

# Task 4: 编译并运行 PROSAC 单元测试（独立程序，仅依赖 vm4_prosac.cpp）
test_prosac: $(PROSAC_TEST)
	$(PROSAC_TEST)

$(PROSAC_TEST): $(TESTDIR)/test_prosac.cpp $(SRCDIR)/vm4_prosac.cpp $(INCDIR)/vm4_prosac.h
	@if not exist $(TESTDIR) mkdir $(TESTDIR)
	$(CXX) -O2 -std=c++17 -Wall -I$(INCDIR) -o $@ $(TESTDIR)/test_prosac.cpp $(SRCDIR)/vm4_prosac.cpp -lm

# Task 5: 仅编译 vm4_bayes.cpp（验证无语法错误）
check_bayes: $(SRCDIR)/vm4_bayes.cpp $(INCDIR)/vm4_bayes.h
	$(CXX) $(CXXFLAGS) -c -o $(SRCDIR)/vm4_bayes.o $< -I$(INCDIR)
	@echo "vm4_bayes.cpp compiled OK"

# Task 5: 编译并运行贝叶斯验证单元测试（独立程序，仅依赖 vm4_bayes.cpp）
test_bayes: $(TEST_BAYES)
	$(TEST_BAYES)

$(TEST_BAYES): $(TESTDIR)/test_bayes.cpp $(SRCDIR)/vm4_bayes.cpp $(INCDIR)/vm4_bayes.h
	@if not exist $(TESTDIR) mkdir $(TESTDIR)
	$(CXX) -O2 -std=c++17 -Wall -I$(INCDIR) -o $@ $(TESTDIR)/test_bayes.cpp $(SRCDIR)/vm4_bayes.cpp -lm

# Task 1: 编译并运行贝叶斯增量验证单元测试
test_bayes_increment: $(TEST_BAYES_INC)
	$(TEST_BAYES_INC)

$(TEST_BAYES_INC): $(TESTDIR)/test_bayes_increment.cpp $(SRCDIR)/vm4_bayes.cpp $(INCDIR)/vm4_bayes.h
	@if not exist $(TESTDIR) mkdir $(TESTDIR)
	$(CXX) -O2 -std=c++17 -Wall -I$(INCDIR) -o $@ $(TESTDIR)/test_bayes_increment.cpp $(SRCDIR)/vm4_bayes.cpp -lm

# Task 6: 仅编译 vm4_triangle.cpp（验证无语法错误）
check_triangle: $(SRCDIR)/vm4_triangle.cpp $(INCDIR)/vm4_triangle.h
	$(CXX) $(CXXFLAGS) -c -o $(SRCDIR)/vm4_triangle.o $< -I$(INCDIR)
	@echo "vm4_triangle.cpp compiled OK"

# Task 6: 编译并运行三角形双特征验证单元测试（独立程序，仅依赖 vm4_triangle.cpp）
test_triangle: $(TEST_TRIANGLE)
	$(TEST_TRIANGLE)

$(TEST_TRIANGLE): $(TESTDIR)/test_triangle.cpp $(SRCDIR)/vm4_triangle.cpp $(INCDIR)/vm4_triangle.h
	@if not exist $(TESTDIR) mkdir $(TESTDIR)
	$(CXX) -O2 -std=c++17 -Wall -I$(INCDIR) -o $@ $(TESTDIR)/test_triangle.cpp $(SRCDIR)/vm4_triangle.cpp -lm

clean:
	del /f /q $(TARGET) 2>nul
	del /f /q $(TEST_TARGET) 2>nul
	del /f /q $(TEST_DENSITY) 2>nul
	del /f /q $(PROSAC_TEST) 2>nul
	del /f /q $(TEST_BAYES) 2>nul
	del /f /q $(TEST_BAYES_INC) 2>nul
	del /f /q $(TEST_TRIANGLE) 2>nul
	del /f /q $(SRCDIR)\*.o 2>nul

.PHONY: all clean check_kvector check_prosac check_bayes check_triangle test test_density test_prosac test_bayes test_bayes_increment test_triangle
