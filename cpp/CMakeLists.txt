# Cross-platform build for the C++ micrograd port.
# Works with MSVC (Visual Studio 2022), GCC and Clang on any OS.
#
#   cmake -S . -B build
#   cmake --build build --config Release
#
cmake_minimum_required(VERSION 3.15)
project(micrograd_cpp LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
    set(CMAKE_BUILD_TYPE Release CACHE STRING "" FORCE)
endif()

add_executable(micrograd main.cpp)   # engine.h and nn.h are header-only

if(MSVC)
    target_compile_options(micrograd PRIVATE /O2)
else()
    target_compile_options(micrograd PRIVATE -O2)
endif()
