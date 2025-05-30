cmake_minimum_required(VERSION 3.18.0)
project (BeeDNN)
enable_testing()

# if eigen is installed, use it to speed up training
if(DEFINED ENV{EIGEN_PATH})
    message(STATUS "Using Eigen at EIGEN_PATH env")
    include_directories($ENV{EIGEN_PATH})
    add_compile_definitions(USE_EIGEN)
else()
    message(STATUS "Not using Eigen (EIGEN_PATH env not defined)")
    message(STATUS "Training will be slow.")
    message(STATUS "To speed up, install Eigen headers from https://eigen.tuxfamily.org/")
endif()

find_package(OpenMP)
if(OPENMP_FOUND)
    set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
    set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} ${OpenMP_EXE_LINKER_FLAGS}")
endif()

add_subdirectory(src)
add_subdirectory(tests)
add_subdirectory(samples)
add_subdirectory(java_binding)

