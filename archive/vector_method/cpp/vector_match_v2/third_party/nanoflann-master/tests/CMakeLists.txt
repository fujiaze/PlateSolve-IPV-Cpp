# Modern CMake minimum requirement
cmake_minimum_required(VERSION 3.14)
project(tests)

# Fix a "bug" in VS11 (MSVC 2012):
if(MSVC)
    add_compile_definitions(_VARIADIC_MAX=10)
endif()

# Find or build GTest
if(NANOFLANN_USE_SYSTEM_GTEST)
    find_package(GTest REQUIRED)
    set(GTEST_TARGET GTest::gtest)
else()
    # EXCLUDE_FROM_ALL prevents gtest from being installed with your project
    add_subdirectory(gtest-1.8.0 EXCLUDE_FROM_ALL)
    
    # Use the target defined by your local gtest
    set(GTEST_TARGET mygtest)

    # Mark local GTest headers as SYSTEM to suppress pedantic warnings from them
    if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
        target_include_directories(${GTEST_TARGET} SYSTEM INTERFACE "gtest-1.8.0/include")
    else()
        target_include_directories(${GTEST_TARGET} INTERFACE "gtest-1.8.0/include")
    endif()
endif()

# Test Project Executable
add_executable(unit_tests test_main.cpp)
set_target_properties(unit_tests PROPERTIES FOLDER "./")

# ----------------------------------------------------------------------------
# STRICT WARNINGS - Scoped specifically to the unit_tests target
# ----------------------------------------------------------------------------
if(MSVC)
    target_compile_options(unit_tests PRIVATE /W4)
else()
    target_compile_options(unit_tests PRIVATE 
        -Wall 
        -Wextra 
        -Wpedantic 
        -Wconversion 
        -Wshadow
    )
endif()

# Standard Cross-Platform Threading
find_package(Threads REQUIRED)

# Link necessary libraries
target_link_libraries(unit_tests 
    PRIVATE 
        ${GTEST_TARGET} 
        nanoflann
        Threads::Threads
)

# Register with CTest
include(CTest)
add_test(NAME unit_tests_run COMMAND unit_tests)