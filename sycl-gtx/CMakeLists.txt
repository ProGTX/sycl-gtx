set(sourceRootPath "${CMAKE_CURRENT_SOURCE_DIR}/source")
set(includeRootPath "${CMAKE_CURRENT_SOURCE_DIR}/include")

get_all_files(sourceList "${sourceRootPath}" "*.cpp")
get_all_files(headerList "${includeRootPath}" "*.h")

add_library(sycl-gtx
  "${sourceList}"
  "${headerList}"
  "${CMAKE_CURRENT_SOURCE_DIR}/include/CL/sycl.hpp"
  "${CMAKE_CURRENT_SOURCE_DIR}/include/CL/sycl_gtx_compatibility.h"
)

include_directories(sycl-gtx "${includeRootPath}")
include_directories(sycl-gtx ${OpenCL_INCLUDE_DIRS})

target_link_libraries(sycl-gtx ${OpenCL_LIBRARIES})

msvc_set_source_filters("${sourceRootPath}" "${sourceList}")
msvc_set_header_filters("${includeRootPath}" "${headerList}")
