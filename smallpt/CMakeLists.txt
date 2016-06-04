GET_ALL_FILES(_source_list "${CMAKE_CURRENT_SOURCE_DIR}" "*.cpp")
GET_ALL_FILES(_header_list "${CMAKE_CURRENT_SOURCE_DIR}" "*.h")

set(_project_name "smallpt")

add_executable(${_project_name} "${_source_list}" "${_header_list}")

include_directories(${_project_name} "${_sycl_gtx_include_path}")
include_directories(${_project_name} ${OpenCL_INCLUDE_DIRS})

target_link_libraries(${_project_name} sycl-gtx)
target_link_libraries(${_project_name} ${OpenCL_LIBRARIES})

MSVC_SET_SOURCE_FILTERS("${CMAKE_CURRENT_SOURCE_DIR}" "${_source_list}")
MSVC_SET_HEADER_FILTERS("${CMAKE_CURRENT_SOURCE_DIR}" "${_header_list}")
