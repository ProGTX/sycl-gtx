GET_ALL_FILES(
  _source_list
  "${CMAKE_CURRENT_SOURCE_DIR}"
  "*.cpp"
  FALSE
)

ADD_TEST_GROUP("regression" "${_source_list}")
