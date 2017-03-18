#pragma once

#include "SYCL/detail/common.h"
#include <map>

namespace cl {
namespace sycl {
namespace detail {

static const int num_errors = 69;
static const char* code_not_exists = "SYCL_ERROR_CODE_DOES_NOT_EXIST";

static const char* errors[num_errors] = {
    "CL_SUCCESS",
    "CL_DEVICE_NOT_FOUND",
    "CL_DEVICE_NOT_AVAILABLE",
    "CL_COMPILER_NOT_AVAILABLE",
    "CL_MEM_OBJECT_ALLOCATION_FAILURE",
    "CL_OUT_OF_RESOURCES",
    "CL_OUT_OF_HOST_MEMORY",
    "CL_PROFILING_INFO_NOT_AVAILABLE",
    "CL_MEM_COPY_OVERLAP",
    "CL_IMAGE_FORMAT_MISMATCH",
    "CL_IMAGE_FORMAT_NOT_SUPPORTED",
    "CL_BUILD_PROGRAM_FAILURE",
    "CL_MAP_FAILURE",
    "CL_MISALIGNED_SUB_BUFFER_OFFSET",
    "CL_EXEC_STATUS_ERROR_FOR_EVENTS_IN_WAIT_LIST",
    "CL_COMPILE_PROGRAM_FAILURE",
    "CL_LINKER_NOT_AVAILABLE",
    "CL_LINK_PROGRAM_FAILURE",
    "CL_DEVICE_PARTITION_FAILED",
    "CL_KERNEL_ARG_INFO_NOT_AVAILABLE",

    code_not_exists,
    code_not_exists,
    code_not_exists,
    code_not_exists,
    code_not_exists,
    code_not_exists,
    code_not_exists,
    code_not_exists,
    code_not_exists,
    code_not_exists,

    "CL_INVALID_VALUE",
    "CL_INVALID_DEVICE_TYPE",
    "CL_INVALID_PLATFORM",
    "CL_INVALID_DEVICE",
    "CL_INVALID_CONTEXT",
    "CL_INVALID_QUEUE_PROPERTIES",
    "CL_INVALID_COMMAND_QUEUE",
    "CL_INVALID_HOST_PTR",
    "CL_INVALID_MEM_OBJECT",
    "CL_INVALID_IMAGE_FORMAT_DESCRIPTOR",
    "CL_INVALID_IMAGE_SIZE",
    "CL_INVALID_SAMPLER",
    "CL_INVALID_BINARY",
    "CL_INVALID_BUILD_OPTIONS",
    "CL_INVALID_PROGRAM",
    "CL_INVALID_PROGRAM_EXECUTABLE",
    "CL_INVALID_KERNEL_NAME",
    "CL_INVALID_KERNEL_DEFINITION",
    "CL_INVALID_KERNEL",
    "CL_INVALID_ARG_INDEX",
    "CL_INVALID_ARG_VALUE",
    "CL_INVALID_ARG_SIZE",
    "CL_INVALID_KERNEL_ARGS",
    "CL_INVALID_WORK_DIMENSION",
    "CL_INVALID_WORK_GROUP_SIZE",
    "CL_INVALID_WORK_ITEM_SIZE",
    "CL_INVALID_GLOBAL_OFFSET",
    "CL_INVALID_EVENT_WAIT_LIST",
    "CL_INVALID_EVENT",
    "CL_INVALID_OPERATION",
    "CL_INVALID_GL_OBJECT",
    "CL_INVALID_BUFFER_SIZE",
    "CL_INVALID_MIP_LEVEL",
    "CL_INVALID_GLOBAL_WORK_SIZE",
    "CL_INVALID_PROPERTY",
    "CL_INVALID_IMAGE_DESCRIPTOR",
    "CL_INVALID_COMPILER_OPTIONS",
    "CL_INVALID_LINKER_OPTIONS",
    "CL_INVALID_DEVICE_PARTITION_COUNT",
};

static const char* error_string(int error_code) {
  if (error_code > 0 || error_code <= -num_errors) {
    return code_not_exists;
  }
  return errors[-error_code];
}

namespace error {

struct code {
  enum value_t : int {
    GENERAL_FAILURE = 1,
    NOT_IN_COMMAND_GROUP_SCOPE,
    TRYING_TO_WRITE_READ_ONLY_BUFFER,
    BUFFER_NOT_INITIALIZED,
    NOT_IN_KERNEL_SCOPE
  };
};

#define SYCL_ADD_ERROR(value) \
  { value, #value }

static const std::map<code::value_t, string_class> codes = {
    SYCL_ADD_ERROR(code::GENERAL_FAILURE),
    SYCL_ADD_ERROR(code::NOT_IN_COMMAND_GROUP_SCOPE),
    SYCL_ADD_ERROR(code::TRYING_TO_WRITE_READ_ONLY_BUFFER),
    SYCL_ADD_ERROR(code::BUFFER_NOT_INITIALIZED),
    SYCL_ADD_ERROR(code::NOT_IN_KERNEL_SCOPE),
};

}  // namespace error

}  // namespace detail
}  // namespace sycl
}  // namespace cl

#undef SYCL_ADD_ERROR
