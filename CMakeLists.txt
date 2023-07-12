cmake_minimum_required(VERSION 3.25)
project(UltraGrad)

set(CMAKE_CXX_STANDARD 20)

include_directories(.)

add_executable(UltraGrad
        #engine.cpp
        engine.hpp
        main.cpp
        #tensor.cpp
        #tensor.hpp
        utils.cpp
        utils.hpp
        smartptr.hpp
        size.hpp
        size.cpp
        deprecated.hpp
        timer.hpp)
