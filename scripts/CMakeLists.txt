cmake_minimum_required(VERSION 3.4)
project(pysteiner)

include_directories(../include)
link_directories(../lib)
add_subdirectory(../externals/pybind11-2.9.2 build)
pybind11_add_module(pysteiner pysteiner.cpp)
target_link_libraries(pysteiner PRIVATE steiner)
