function(GET_ALL_FILES
  _source_list_out
  _root_path
  _regex
  # ARGV3 _recursion = TRUE
)
  set(_glob GLOB_RECURSE)
  if((${ARGC} GREATER 3) AND ("${ARGV3}" STREQUAL "FALSE"))
    set(_glob GLOB)
  endif()
  file(
    ${_glob} _source_list_local
    LIST_DIRECTORIES false
    "${_root_path}/${_regex}"
  )
  set(${_source_list_out} ${_source_list_local} PARENT_SCOPE)
endfunction()

function(MSVC_SET_FILTERS _source_root_path _source_list _prefix)
  # http://stackoverflow.com/a/33813154
  foreach(_source IN ITEMS ${_source_list})
    get_filename_component(_source_path "${_source}" PATH)
    file(
      RELATIVE_PATH _source_path_rel
      "${_source_root_path}" "${_source_path}"
    )
    string(REPLACE "/" "\\" _group_path "${_source_path_rel}")
    source_group("${_prefix}${_group_path}" FILES "${_source}")
  endforeach()
endfunction()
  
function(MSVC_SET_SOURCE_FILTERS _source_root_path _source_list)
  MSVC_SET_FILTERS("${_source_root_path}" "${_source_list}" "Source Files\\")
endfunction()
  
function(MSVC_SET_HEADER_FILTERS _include_root_path _header_list)
  MSVC_SET_FILTERS("${_include_root_path}" "${_header_list}" "Header Files\\")
endfunction()

function(ADD_TEST_GROUP _group _source_list)
  foreach(_test ${_source_list})
    get_filename_component(_test "${_test}" NAME)
    get_filename_component(_project_name "${_test}" NAME_WE)
    
    add_executable(${_project_name} ${_test})
    
    include_directories(${_project_name} "${_sycl_gtx_include_path}")
    include_directories(${_project_name} ${OpenCL_INCLUDE_DIRS})
    
    target_link_libraries(${_project_name} sycl-gtx)
    target_link_libraries(${_project_name} ${OpenCL_LIBRARIES})
    
    if(MSVC)
      set_target_properties(
        ${_project_name} PROPERTIES FOLDER "tests/${_group}"
      )
    endif(MSVC)
    
    add_test(NAME ${_project_name} COMMAND ${_project_name})
  endforeach()
endfunction()
