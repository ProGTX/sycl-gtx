cmake_minimum_required(VERSION 2.8)

project(sycl-gtx-solution)

# Requirements
set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
find_package(OpenCL REQUIRED)

# Common functions
include("cmake/common.cmake")

if(MSVC)
  set_property(GLOBAL PROPERTY USE_FOLDERS ON)
endif()

set(_sycl_gtx_include_path "${CMAKE_SOURCE_DIR}/sycl-gtx/include")

# Main library
add_subdirectory(sycl-gtx)

# Tests
enable_testing()
add_subdirectory(tests)

# Other projects
add_subdirectory(smallpt)
