cmake_minimum_required(VERSION 3.28)

################
# CUDA PROJECT #
################

project(n-body-cuda CUDA)

set(CMAKE_CUDA_STANDARD 20)

include_directories(src/cuda/include)

file(GLOB N_BODY_CUDA_SOURCES
        "src/cuda/**/*.cpp"
        "src/cuda/**/*.cu"
        "src/cuda/main.cpp"
)

add_executable(n-body-cuda ${N_BODY_CUDA_SOURCES})

set_target_properties(n-body-cuda PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON)

##################
# OpenMP PROJECT #
##################

project(n-body-openMP CXX)

find_package(OpenMP REQUIRED)

include_directories(/u/sw/toolchains/gcc-glibc/11.2.0/base/include
        /u/sw/toolchains/gcc-glibc/11.2.0/pkgs/eigen/3.3.9/include/eigen3)

file(GLOB SRC_FILES "src/openmp/*.cpp" "src/openmp/Solver/*.cpp")
add_executable(main-omp ${SRC_FILES})

target_compile_options(main-omp PRIVATE -fopenmp)
target_link_libraries(main-omp PUBLIC OpenMP::OpenMP_CXX)

##################
# OpenGL PROJECT #
##################

project(n-body-openGL CXX)

find_package(OpenGL REQUIRED)
find_package(glfw3 REQUIRED)

file(GLOB SRC_FILES "src/opengl/*.cpp" "src/openmp/Vector.cpp")
add_executable(opengl ${SRC_FILES})

target_link_libraries(opengl PUBLIC OpenGL::GL glfw GLEW GLU)