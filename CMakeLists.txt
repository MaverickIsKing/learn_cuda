cmake_minimum_required(VERSION 3.20.0)
project(learn_cuda VERSION 0.1.0 LANGUAGES CUDA CXX C)
find_package(CUDAToolkit)
add_subdirectory(1_cuda_reduce_study)