cmake_minimum_required(VERSION 2.8)

project(sycl-gtx-solution)

# CMake variable to string resolution policy
cmake_policy(SET CMP0054 NEW)

# Requirements
set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
find_package(OpenCL REQUIRED)

# Common functions
set(_cmake_files
  "cmake/common.cmake"
  "cmake/color_diagnostics.cmake"
)
foreach(_file ${_cmake_files})
  include(${_file})
endforeach()

if(MSVC)
  set_property(GLOBAL PROPERTY USE_FOLDERS ON)
endif()

set(SYCL_GTX_INCLUDE_PATH ${CMAKE_SOURCE_DIR}/sycl-gtx/include)

# Main library
add_subdirectory(sycl-gtx)

# Tests
enable_testing()
add_subdirectory(tests)

# Other projects
add_subdirectory(smallpt)
