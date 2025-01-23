cmake_minimum_required(VERSION 3.16)

# path to clang compiler
set(CMAKE_C_COMPILER "/opt/homebrew/opt/llvm/bin/clang")
#path to clang++ compiler
set(CMAKE_CXX_COMPILER "/opt/homebrew/opt/llvm/bin/clang++")

project(NBody)

set(CMAKE_CXX_STANDARD 20)

#OPEN MP flags
set(OpenMP_CXX_FLAGS "-Xpreprocessor -fopenmp")
set(OpenMP_CXX_LIB_NAMES "omp")
#OPEN MP flags
set(OpenMP_C_FLAGS "-Xpreprocessor -fopenmp")
set(OpenMP_C_LIB_NAMES "omp")

#Path to libomp.dylib file
set(OpenMP_omp_LIBRARY "/opt/homebrew/Cellar/libomp/19.1.6/lib/libomp.dylib")

# Find required packages: OpenMP, OpenGL and glfw3
find_package(OpenMP REQUIRED)

find_package(OpenGL REQUIRED)

find_package(glfw3 REQUIRED)

include_directories(
        /opt/homebrew/opt/glew/include #path to glew/include
        /opt/homebrew/opt/glfw/include #path to glfw/include
)
link_directories(/opt/homebrew/opt/glew/lib /opt/homebrew/opt/glfw/lib)

#OPENMP EXECUTABLE
file(GLOB SRC_FILES "src/openmp/*.cpp" "src/openmp/Solver/*.cpp" )
add_executable(openmp_main ${SRC_FILES})

target_link_libraries(openmp_main PUBLIC OpenMP::OpenMP_CXX)

target_compile_options(openmp_main PRIVATE -fopenmp)

#OPENGL EXECUTABLE
file(GLOB SRC_FILES "src/opengl/*.cpp" "src/openmp/Vector.cpp" )
add_executable(opengl_main ${SRC_FILES})

target_link_libraries(opengl_main PUBLIC OpenGL::GL glfw GLEW)

# Testing with GoogleTest
include(FetchContent)
FetchContent_Declare(
        googletest
        URL https://github.com/google/googletest/archive/03597a01ee50ed33e9dfd640b249b4be3799d395.zip
)
set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)
FetchContent_MakeAvailable(googletest)

enable_testing()
file(GLOB TEST_FILES "test/openmp/*.cpp" "src/openmp/Solver/*.cpp" "src/openmp/Vector.cpp")

add_executable(
        openmp_test
        ${TEST_FILES}
)
target_link_libraries(
        openmp_test
        GTest::gtest_main
)

include(GoogleTest)
gtest_discover_tests(openmp_test)