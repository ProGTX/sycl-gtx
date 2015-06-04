#pragma once

#include "info.h"
#include "../common.h"
#include <CL/cl.h>

namespace cl {
namespace sycl {

// Forward declaration
template <typename EnumClass, EnumClass Value>
struct param_traits2;

namespace detail {

template <typename EnumClass, EnumClass Value, typename ReturnType, typename CLType, typename CLReturnType = ReturnType>
struct param_traits_helper {
	using type = ReturnType;
	using cl_flag_type = CLType;
	using cl_return_type = CLReturnType;
};

template <typename Contained>
struct traits {
	using return_t = vector_class<Contained>;
	static const int BUFFER_SIZE = 1024;
	static const size_t type_size = sizeof(Contained);
};
template <>
struct traits<string_class> {
	using return_t = string_class;
	static const int BUFFER_SIZE = 8192;
	static const size_t type_size = sizeof(char);
};

template <typename cl_input_t>
using opencl_info_f = cl_int(CL_API_CALL*)(cl_input_t, cl_uint, size_t, void*, size_t*);

template <typename cl_input_t, opencl_info_f<cl_input_t> F>
struct info_function_helper {
	template <class... Args>
	static cl_int get(Args... args) {
		return F(args...);
	}
};

template <typename EnumClass>
struct info_function;
template <>
struct info_function<info::context> : info_function_helper<cl_context, clGetContextInfo>{};
template <>
struct info_function<info::device> : info_function_helper<cl_device_id, clGetDeviceInfo>{};

template <typename EnumClass, EnumClass param, size_t size, typename cl_input_t>
static void get_cl_info(cl_input_t data_ptr, void* param_value, size_t* actual_size = nullptr) {
	auto error_code = info_function<EnumClass>::get(
		data_ptr, (param_traits2<EnumClass, param>::cl_flag_type)param, size, param_value, actual_size
	);
	detail::error::report(error_code);
}

} // namespace detail


#define SYCL_ADD_TRAIT_DIFF_RET(EnumClass, Value, ReturnType, CLType, CLReturnType)	\
template <>																					\
struct param_traits2<EnumClass, Value>														\
	: detail::param_traits_helper<															\
		EnumClass, Value, ReturnType, CLType, CLReturnType									\
	> {};

#define SYCL_ADD_TRAIT(EnumClass, Value, ReturnType, CLType)	\
	SYCL_ADD_TRAIT_DIFF_RET(EnumClass, Value, ReturnType, CLType, ReturnType)

// 3.3.3.2 Context information descriptors
// https://www.khronos.org/registry/cl/sdk/1.2/docs/man/xhtml/clGetContextInfo.html

#define SYCL_ADD_CONTEXT_TRAIT(Value, ReturnType)	\
	SYCL_ADD_TRAIT(info::context, Value, ReturnType, cl_context_info)

SYCL_ADD_CONTEXT_TRAIT(info::context::reference_count,	cl_uint)
SYCL_ADD_CONTEXT_TRAIT(info::context::num_devices,		cl_uint)
SYCL_ADD_CONTEXT_TRAIT(info::context::devices,			vector_class<cl_device_id>)
SYCL_ADD_CONTEXT_TRAIT(info::context::gl_interop,		info::gl_context_interop)

#undef SYCL_ADD_CONTEXT_TRAIT


// 3.3.2.1 Platform information descriptors
// https://www.khronos.org/registry/cl/sdk/1.2/docs/man/xhtml/clGetPlatformInfo.html

#define SYCL_ADD_PLATFORM_TRAIT(Value)	\
	SYCL_ADD_TRAIT(info::platform, Value, string_class, cl_platform_info)

SYCL_ADD_PLATFORM_TRAIT(info::platform::profile)
SYCL_ADD_PLATFORM_TRAIT(info::platform::version)
SYCL_ADD_PLATFORM_TRAIT(info::platform::name)
SYCL_ADD_PLATFORM_TRAIT(info::platform::vendor)
SYCL_ADD_PLATFORM_TRAIT(info::platform::extensions)

#undef SYCL_ADD_PLATFORM_TRAIT


// 3.3.4.2 Device information descriptors
// https://www.khronos.org/registry/cl/sdk/1.2/docs/man/xhtml/clGetDeviceInfo.html

#define SYCL_ADD_DEVICE_TRAIT_DIFF_RET(Value, ReturnType, CLReturnType)	\
	SYCL_ADD_TRAIT_DIFF_RET(info::device, Value, ReturnType, cl_device_info, CLReturnType)
#define SYCL_ADD_DEVICE_TRAIT(Value, ReturnType)	\
	SYCL_ADD_TRAIT(info::device, Value, ReturnType, cl_device_info)

// Forward declaration
template <int dimensions>
struct id;

SYCL_ADD_DEVICE_TRAIT_DIFF_RET(info::device::device_type,				info::device_type, cl_device_type)
SYCL_ADD_DEVICE_TRAIT(info::device::vendor_id,							cl_uint)

SYCL_ADD_DEVICE_TRAIT(info::device::max_compute_units,					cl_uint)
SYCL_ADD_DEVICE_TRAIT(info::device::max_work_item_dimensions,			cl_uint)
SYCL_ADD_DEVICE_TRAIT_DIFF_RET(info::device::max_work_item_sizes,		id<3>, size_t[])
SYCL_ADD_DEVICE_TRAIT(info::device::max_work_group_size,				size_t)

SYCL_ADD_DEVICE_TRAIT(info::device::preferred_vector_width_char,		cl_uint)
SYCL_ADD_DEVICE_TRAIT(info::device::preferred_vector_width_short,		cl_uint)
SYCL_ADD_DEVICE_TRAIT(info::device::preferred_vector_width_int,			cl_uint)
SYCL_ADD_DEVICE_TRAIT(info::device::preferred_vector_width_long_long,	cl_uint)
SYCL_ADD_DEVICE_TRAIT(info::device::preferred_vector_width_float,		cl_uint)
SYCL_ADD_DEVICE_TRAIT(info::device::preferred_vector_width_double,		cl_uint)
SYCL_ADD_DEVICE_TRAIT(info::device::preferred_vector_width_half,		cl_uint)

SYCL_ADD_DEVICE_TRAIT(info::device::native_vector_width_char,			cl_uint)
SYCL_ADD_DEVICE_TRAIT(info::device::native_vector_width_short,			cl_uint)
SYCL_ADD_DEVICE_TRAIT(info::device::native_vector_width_int,			cl_uint)
SYCL_ADD_DEVICE_TRAIT(info::device::native_vector_width_long_long,		cl_uint)
SYCL_ADD_DEVICE_TRAIT(info::device::native_vector_width_float,			cl_uint)
SYCL_ADD_DEVICE_TRAIT(info::device::native_vector_width_double,			cl_uint)
SYCL_ADD_DEVICE_TRAIT(info::device::native_vector_width_half,			cl_uint)

SYCL_ADD_DEVICE_TRAIT(info::device::max_clock_frequency,				cl_uint)
SYCL_ADD_DEVICE_TRAIT(info::device::address_bits,						cl_uint)
SYCL_ADD_DEVICE_TRAIT(info::device::max_mem_alloc_size,					cl_ulong)

SYCL_ADD_DEVICE_TRAIT(info::device::image_support,						cl_bool)
SYCL_ADD_DEVICE_TRAIT(info::device::max_read_image_args,				cl_uint)
SYCL_ADD_DEVICE_TRAIT(info::device::max_write_image_args,				cl_uint)
SYCL_ADD_DEVICE_TRAIT(info::device::image2d_max_height,					size_t)
SYCL_ADD_DEVICE_TRAIT(info::device::image2d_max_width,					size_t)
SYCL_ADD_DEVICE_TRAIT(info::device::image3d_max_height,					size_t)
SYCL_ADD_DEVICE_TRAIT(info::device::image3d_max_width,					size_t)
SYCL_ADD_DEVICE_TRAIT(info::device::image3d_max_depth,					size_t)
SYCL_ADD_DEVICE_TRAIT(info::device::image_max_buffer_size,				size_t)
SYCL_ADD_DEVICE_TRAIT(info::device::image_max_array_size,				size_t)

SYCL_ADD_DEVICE_TRAIT(info::device::max_samplers,						cl_uint)
SYCL_ADD_DEVICE_TRAIT(info::device::max_parameter_size,					size_t)
SYCL_ADD_DEVICE_TRAIT(info::device::mem_base_addr_align,				cl_uint)

SYCL_ADD_DEVICE_TRAIT_DIFF_RET(info::device::single_fp_config,			info::device_fp_config, cl_device_fp_config)
SYCL_ADD_DEVICE_TRAIT_DIFF_RET(info::device::double_fp_config,			info::device_fp_config, cl_device_fp_config)

SYCL_ADD_DEVICE_TRAIT_DIFF_RET(info::device::global_mem_cache_type,		info::global_mem_cache_type, cl_device_mem_cache_type)
SYCL_ADD_DEVICE_TRAIT(info::device::global_mem_cache_line_size,			cl_uint)
SYCL_ADD_DEVICE_TRAIT(info::device::global_mem_cache_size,				cl_ulong)
SYCL_ADD_DEVICE_TRAIT(info::device::global_mem_size,					cl_ulong)
SYCL_ADD_DEVICE_TRAIT(info::device::max_constant_buffer_size,			cl_ulong)
SYCL_ADD_DEVICE_TRAIT(info::device::max_constant_args,					cl_uint)
SYCL_ADD_DEVICE_TRAIT_DIFF_RET(info::device::local_mem_type,			info::local_mem_type, cl_device_local_mem_type)
SYCL_ADD_DEVICE_TRAIT(info::device::local_mem_size,						cl_ulong)

SYCL_ADD_DEVICE_TRAIT(info::device::error_correction_support,			cl_bool)
SYCL_ADD_DEVICE_TRAIT(info::device::host_unified_memory,				cl_bool)
SYCL_ADD_DEVICE_TRAIT(info::device::profiling_timer_resolution,			size_t)
SYCL_ADD_DEVICE_TRAIT(info::device::endian_little,						cl_bool)
SYCL_ADD_DEVICE_TRAIT(info::device::is_available,						cl_bool)
SYCL_ADD_DEVICE_TRAIT(info::device::is_compiler_available,				cl_bool)
SYCL_ADD_DEVICE_TRAIT(info::device::is_linker_available,				cl_bool)

SYCL_ADD_DEVICE_TRAIT_DIFF_RET(info::device::execution_capabilities,	info::device_exec_capabilities, cl_device_exec_capabilities)
SYCL_ADD_DEVICE_TRAIT_DIFF_RET(info::device::queue_properties,			vector_class<info::device_queue_properties>, cl_command_queue_properties)
SYCL_ADD_DEVICE_TRAIT(info::device::built_in_kernels,					string_class)
SYCL_ADD_DEVICE_TRAIT(info::device::platform,							cl_platform_id)
SYCL_ADD_DEVICE_TRAIT(info::device::name,								string_class)
SYCL_ADD_DEVICE_TRAIT(info::device::vendor,								string_class)
SYCL_ADD_DEVICE_TRAIT(info::device::driver_version,						string_class)
SYCL_ADD_DEVICE_TRAIT(info::device::profile,							string_class)
SYCL_ADD_DEVICE_TRAIT(info::device::device_version,						string_class)
SYCL_ADD_DEVICE_TRAIT(info::device::opencl_version,						string_class)
SYCL_ADD_DEVICE_TRAIT(info::device::extensions,							string_class)

SYCL_ADD_DEVICE_TRAIT(info::device::printf_buffer_size,					size_t)
SYCL_ADD_DEVICE_TRAIT(info::device::preferred_interop_user_sync,		cl_bool)
SYCL_ADD_DEVICE_TRAIT(info::device::parent_device,						cl_device_id)
SYCL_ADD_DEVICE_TRAIT(info::device::partition_max_sub_devices,			cl_uint)
SYCL_ADD_DEVICE_TRAIT_DIFF_RET(info::device::partition_properties,		vector_class<info::device_partition_property>, cl_device_partition_property[])
SYCL_ADD_DEVICE_TRAIT_DIFF_RET(info::device::partition_affinity_domain,	info::device_affinity_domain, cl_device_affinity_domain)
SYCL_ADD_DEVICE_TRAIT_DIFF_RET(info::device::partition_type,			vector_class<info::device_partition_type>, cl_device_partition_property[])
SYCL_ADD_DEVICE_TRAIT(info::device::reference_count,					cl_uint)

#undef SYCL_ADD_DEVICE_TRAIT


#undef SYCL_ADD_TRAIT

} // namespace sycl
} // namespace cl
