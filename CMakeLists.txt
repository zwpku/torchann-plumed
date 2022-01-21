cmake_minimum_required(VERSION 3.0 FATAL_ERROR)
project(TorchANN-Plumed)

find_package(Torch REQUIRED)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${TORCH_CXX_FLAGS}")

find_library(PLUMED plumedKernel)
add_library(TorchANNPlumed SHARED TorchANN.cpp)
find_path(PLUMED_INC_DIR function/Function.h)

target_include_directories(TorchANNPlumed PUBLIC "${PLUMED_INC_DIR}")
target_link_libraries(TorchANNPlumed "${TORCH_LIBRARIES}")
target_link_libraries(TorchANNPlumed "${PLUMED}/libplumedKernel.so")
set_property(TARGET TorchANNPlumed PROPERTY CXX_STANDARD 14)

