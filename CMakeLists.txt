cmake_minimum_required(VERSION 3.2)

project(sycl-gtx-solution)

# Requirements
set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
find_package(OpenCL REQUIRED)

# Common functions
set(SYCL_GTX_CMAKE_FILES "cmake/common.cmake" "cmake/color_diagnostics.cmake")
foreach(fileName ${SYCL_GTX_CMAKE_FILES})
  include(${fileName})
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
