get_all_files(sourceList "${CMAKE_CURRENT_SOURCE_DIR}" "*.cpp")
get_all_files(headerList "${CMAKE_CURRENT_SOURCE_DIR}" "*.h")

set(projectName "smallpt")

add_executable(${projectName} "${sourceList}" "${headerList}")

find_package(OpenMP)
if(OPENMP_FOUND)
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
endif()

include_directories(${projectName} ${SYCL_GTX_INCLUDE_PATH})
include_directories(${projectName} ${OpenCL_INCLUDE_DIRS})

target_link_libraries(${projectName} sycl-gtx)
target_link_libraries(${projectName} ${OpenCL_LIBRARIES})

if(MSVC)
  msvc_set_source_filters("${CMAKE_CURRENT_SOURCE_DIR}" "${sourceList}")
  msvc_set_header_filters("${CMAKE_CURRENT_SOURCE_DIR}" "${headerList}")
endif()
