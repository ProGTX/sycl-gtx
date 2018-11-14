function(get_all_files
         sourceListOut
         rootPath
         regex
         # ARGV3 recursion = TRUE
         )
  set(globList GLOB_RECURSE)
  if((${ARGC} GREATER 3) AND ("${ARGV3}" STREQUAL "FALSE"))
    set(globList GLOB)
  endif()
  file(${globList}
       sourceListLocal
       LIST_DIRECTORIES
       false
       "${rootPath}/${regex}")
  set(${sourceListOut} ${sourceListLocal} PARENT_SCOPE)
endfunction(get_all_files)

function(msvc_set_filters sourceRootPath sourceList prefix)
  # http://stackoverflow.com/a/33813154
  foreach(sourceFile IN ITEMS ${sourceList})
    get_filename_component(sourcePath "${sourceFile}" PATH)
    file(RELATIVE_PATH sourcePathRel "${sourceRootPath}" "${sourcePath}")
    string(REPLACE "/"
                   "\\"
                   groupPath
                   "${sourcePathRel}")
    source_group("${prefix}${groupPath}" FILES "${sourceFile}")
  endforeach()
endfunction(msvc_set_filters)

function(msvc_set_source_filters sourceRootPath sourceList)
  msvc_set_filters("${sourceRootPath}" "${sourceList}" "Source Files\\")
endfunction()

function(msvc_set_header_filters includeRootPath headerList)
  msvc_set_filters("${includeRootPath}" "${headerList}" "Header Files\\")
endfunction()

function(add_test_group groupName sourceList)
  set(groupSet "")
  foreach(testName ${sourceList})
    get_filename_component(testName "${testName}" NAME)
    get_filename_component(projectName "${testName}" NAME_WE)

    add_executable(${projectName} ${testName})
    set(groupSet ${groupSet} ${projectName})

    include_directories(${projectName} ${SYCL_GTX_INCLUDE_PATH})
    include_directories(${projectName} ${OpenCL_INCLUDE_DIRS})

    target_link_libraries(${projectName} sycl-gtx)
    target_link_libraries(${projectName} ${OpenCL_LIBRARIES})

    if(MSVC)
      set_target_properties(${projectName}
                            PROPERTIES FOLDER "tests/${groupName}")
    endif(MSVC)

    add_test(NAME ${projectName} COMMAND ${projectName})
  endforeach(testName)

  add_custom_target(${groupName}_tests DEPENDS ${groupSet})
  if(MSVC)
    set_target_properties(${groupName}_tests PROPERTIES FOLDER "tests")
  endif(MSVC)
endfunction(add_test_group)
