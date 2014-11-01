#pragma once

#include "common.h"

namespace cl {
namespace sycl {

namespace helper {

template<bool>
struct select_type;

// One cannot safely return char[] from a function
//using char_array = char[];
using char_array = STRING_CLASS;

} // namespace helper

template<typename cl_struct, cl_int name, typename = helper::select_type<true>>
struct param_traits;

#define SYCL_ADD_TRAIT(cl_struct, output_type, condition)	\
template<cl_int name>										\
struct param_traits<										\
	cl_struct,												\
	name,													\
	helper::select_type<(condition)>						\
> {															\
	using param_type = output_type;							\
}

// https://www.khronos.org/registry/cl/sdk/1.0/docs/man/xhtml/clGetPlatformInfo.html
SYCL_ADD_TRAIT(cl_platform_info, helper::char_array, (name >= CL_PLATFORM_PROFILE && name <= CL_PLATFORM_EXTENSIONS));


// https://www.khronos.org/registry/cl/sdk/1.0/docs/man/xhtml/clGetDeviceInfo.html

SYCL_ADD_TRAIT(cl_device_info, cl_uint, (
	name == CL_DEVICE_ADDRESS_BITS				||
	name == CL_DEVICE_GLOBAL_MEM_CACHELINE_SIZE	||
	name == CL_DEVICE_MAX_CLOCK_FREQUENCY		||
	name == CL_DEVICE_MAX_COMPUTE_UNITS			||
	name == CL_DEVICE_MAX_CONSTANT_ARGS			||
	name == CL_DEVICE_MAX_READ_IMAGE_ARGS		||
	name == CL_DEVICE_MAX_SAMPLERS				||
	name == CL_DEVICE_MAX_WORK_ITEM_DIMENSIONS	||
	name == CL_DEVICE_MAX_WRITE_IMAGE_ARGS		||
	name == CL_DEVICE_MEM_BASE_ADDR_ALIGN		||
	name == CL_DEVICE_MIN_DATA_TYPE_ALIGN_SIZE	||
	name == CL_DEVICE_VENDOR_ID					||
	(name >= CL_DEVICE_PREFERRED_VECTOR_WIDTH_CHAR && name <= CL_DEVICE_PREFERRED_VECTOR_WIDTH_DOUBLE)
));

SYCL_ADD_TRAIT(cl_device_info, cl_bool, (
	name == CL_DEVICE_AVAILABLE				||
	name == CL_DEVICE_COMPILER_AVAILABLE	||
	name == CL_DEVICE_ENDIAN_LITTLE			||
	name == CL_DEVICE_IMAGE_SUPPORT			||
	name == CL_DEVICE_ERROR_CORRECTION_SUPPORT
));

SYCL_ADD_TRAIT(cl_device_info, cl_device_fp_config, (
	name == CL_DEVICE_DOUBLE_FP_CONFIG	||
	//name == CL_DEVICE_HALF_FP_CONFIG	||
	name == CL_DEVICE_SINGLE_FP_CONFIG
));

SYCL_ADD_TRAIT(cl_device_info, helper::char_array, (
	name == CL_DEVICE_NAME		||
	name == CL_DEVICE_PROFILE	||
	name == CL_DEVICE_VENDOR	||
	name == CL_DEVICE_VERSION	||
	name == CL_DRIVER_VERSION	||
	name == CL_DEVICE_EXTENSIONS
));

SYCL_ADD_TRAIT(cl_device_info, cl_ulong, (
	name == CL_DEVICE_GLOBAL_MEM_CACHE_SIZE	||
	name == CL_DEVICE_GLOBAL_MEM_SIZE		||
	name == CL_DEVICE_LOCAL_MEM_SIZE		||
	name == CL_DEVICE_MAX_MEM_ALLOC_SIZE	||
	name == CL_DEVICE_MAX_CONSTANT_BUFFER_SIZE
));

SYCL_ADD_TRAIT(cl_device_info, size_t, (
	name == CL_DEVICE_MAX_PARAMETER_SIZE			||
	name == CL_DEVICE_MAX_WORK_GROUP_SIZE			||
	name == CL_DEVICE_MAX_WORK_ITEM_SIZES			||
	name == CL_DEVICE_PROFILING_TIMER_RESOLUTION	||
	(name >= CL_DEVICE_IMAGE2D_MAX_WIDTH && name <= CL_DEVICE_IMAGE3D_MAX_DEPTH)
));

SYCL_ADD_TRAIT(cl_device_info, cl_device_fp_config,			(name == CL_DEVICE_EXECUTION_CAPABILITIES));
SYCL_ADD_TRAIT(cl_device_info, cl_device_mem_cache_type,	(name == CL_DEVICE_GLOBAL_MEM_CACHE_TYPE));
SYCL_ADD_TRAIT(cl_device_info, cl_device_local_mem_type,	(name == CL_DEVICE_LOCAL_MEM_TYPE));
SYCL_ADD_TRAIT(cl_device_info, size_t[],					(name == CL_DEVICE_MAX_WORK_ITEM_SIZES));
SYCL_ADD_TRAIT(cl_device_info, cl_platform_id,				(name == CL_DEVICE_PLATFORM));
SYCL_ADD_TRAIT(cl_device_info, cl_command_queue_properties,	(name == CL_DEVICE_QUEUE_PROPERTIES));
SYCL_ADD_TRAIT(cl_device_info, cl_device_type,				(name == CL_DEVICE_TYPE));

// https://www.khronos.org/registry/cl/sdk/1.0/docs/man/xhtml/clGetContextInfo.html
SYCL_ADD_TRAIT(cl_context_info, cl_uint,					(name == CL_CONTEXT_REFERENCE_COUNT));
SYCL_ADD_TRAIT(cl_context_info, cl_device_id[],				(name == CL_CONTEXT_DEVICES));
SYCL_ADD_TRAIT(cl_context_info, cl_context_properties[],	(name == CL_CONTEXT_PROPERTIES));

} // namespace sycl
} // namespace cl

#undef SYCL_ADD_TRAIT
