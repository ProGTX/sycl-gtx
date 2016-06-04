set(_group "regression")

GET_ALL_FILES(
  _source_list
  "${CMAKE_CURRENT_SOURCE_DIR}"
  "*.cpp"
  FALSE
)

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
  
  # TODO: Does not seem to work
  add_test(NAME ${_project_name} COMMAND ${_project_name})
endforeach()
