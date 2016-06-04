set(_source_root_path "${CMAKE_CURRENT_SOURCE_DIR}/source")
set(_include_root_path "${CMAKE_CURRENT_SOURCE_DIR}/include/SYCL/implementation")

GET_ALL_FILES(_source_list "${_source_root_path}" "*.cpp")
GET_ALL_FILES(_header_list "${_include_root_path}" "*.h")

add_library(sycl-gtx "${_source_list}" "${_header_list}")

include_directories(sycl-gtx "${_include_root_path}")
include_directories(sycl-gtx ${OpenCL_INCLUDE_DIRS})

target_link_libraries(sycl-gtx ${OpenCL_LIBRARIES})

MSVC_SET_SOURCE_FILTERS("${_source_root_path}" "${_source_list}")
MSVC_SET_HEADER_FILTERS("${_include_root_path}" "${_header_list}")
