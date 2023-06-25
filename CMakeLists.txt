cmake_minimum_required(VERSION 3.25)
project(UltraGrad)

set(CMAKE_CXX_STANDARD 17)

include_directories(.)

add_executable(UltraGrad
        engine.cpp
        engine.hpp
        main.cpp
        tensor.cpp
        tensor.hpp
        utils.cpp
        utils.hpp)
