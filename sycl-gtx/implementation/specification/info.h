#pragma once

// C. Interface of Memory Object Information Descriptors

#include <CL/cl.h>
#include <type_traits>

namespace cl {
namespace sycl {
namespace info {

// C.1 Platform Information Descriptors
enum class platform : cl_platform_info {
	profile		= CL_PLATFORM_PROFILE,
	version		= CL_PLATFORM_VERSION,
	name		= CL_PLATFORM_NAME,
	vendor		= CL_PLATFORM_VENDOR,
	extensions	= CL_PLATFORM_EXTENSIONS
};


// C.2 Context Information Descriptors
using gl_context_interop = cl_bool;
enum class context : cl_context_info {
	reference_count	= CL_CONTEXT_REFERENCE_COUNT,
	num_devices		= CL_CONTEXT_NUM_DEVICES,
	devices			= CL_CONTEXT_DEVICES,
	gl_interop		= CL_CONTEXT_INTEROP_USER_SYNC	// TODO: Not sure
};


// C.3 Device Information Descriptors

using device_fp_config			= cl_device_fp_config;
using device_exec_capabilities	= cl_device_exec_capabilities;
using device_queue_properties	= cl_command_queue_properties;

// TODO: Host
enum class device_type : cl_device_type {
	cpu			= CL_DEVICE_TYPE_CPU,
	gpu			= CL_DEVICE_TYPE_GPU,
	accelerator	= CL_DEVICE_TYPE_ACCELERATOR,
	custom		= CL_DEVICE_TYPE_CUSTOM,
	defaults	= CL_DEVICE_TYPE_DEFAULT,
	host,
	all			= CL_DEVICE_TYPE_ALL
};

enum class device : cl_device_info {
	device_type							= CL_DEVICE_TYPE,
	vendor_id							= CL_DEVICE_VENDOR_ID,
	max_compute_units					= CL_DEVICE_MAX_COMPUTE_UNITS,
	max_work_item_dimensions			= CL_DEVICE_MAX_WORK_ITEM_DIMENSIONS,
	max_work_item_sizes					= CL_DEVICE_MAX_WORK_ITEM_SIZES,
	max_work_group_size					= CL_DEVICE_MAX_WORK_GROUP_SIZE,
	preferred_vector_width_char			= CL_DEVICE_PREFERRED_VECTOR_WIDTH_CHAR,
	preferred_vector_width_short		= CL_DEVICE_PREFERRED_VECTOR_WIDTH_SHORT,
	preferred_vector_width_int			= CL_DEVICE_PREFERRED_VECTOR_WIDTH_INT,
	preferred_vector_width_long_long	= CL_DEVICE_PREFERRED_VECTOR_WIDTH_LONG,
	preferred_vector_width_float		= CL_DEVICE_PREFERRED_VECTOR_WIDTH_FLOAT,
	preferred_vector_width_double		= CL_DEVICE_PREFERRED_VECTOR_WIDTH_DOUBLE,
	preferred_vector_width_half			= CL_DEVICE_PREFERRED_VECTOR_WIDTH_HALF,
	native_vector_width_char			= CL_DEVICE_NATIVE_VECTOR_WIDTH_CHAR,
	native_vector_width_short			= CL_DEVICE_NATIVE_VECTOR_WIDTH_SHORT,
	native_vector_width_int				= CL_DEVICE_NATIVE_VECTOR_WIDTH_INT,
	native_vector_width_long_long		= CL_DEVICE_NATIVE_VECTOR_WIDTH_LONG,
	native_vector_width_float			= CL_DEVICE_NATIVE_VECTOR_WIDTH_FLOAT,
	native_vector_width_double			= CL_DEVICE_NATIVE_VECTOR_WIDTH_DOUBLE,
	native_vector_width_half			= CL_DEVICE_NATIVE_VECTOR_WIDTH_HALF,
	max_clock_frequency					= CL_DEVICE_MAX_CLOCK_FREQUENCY,
	address_bits						= CL_DEVICE_ADDRESS_BITS,
	max_mem_alloc_size					= CL_DEVICE_MAX_MEM_ALLOC_SIZE,
	image_support						= CL_DEVICE_IMAGE_SUPPORT,
	max_read_image_args					= CL_DEVICE_MAX_READ_IMAGE_ARGS,
	max_write_image_args				= CL_DEVICE_MAX_WRITE_IMAGE_ARGS,
	image2d_max_height					= CL_DEVICE_IMAGE2D_MAX_HEIGHT,
	image2d_max_width					= CL_DEVICE_IMAGE2D_MAX_WIDTH,
	image3d_max_height					= CL_DEVICE_IMAGE3D_MAX_HEIGHT,
	image3d_max_width					= CL_DEVICE_IMAGE3D_MAX_WIDTH,
	image3d_max_depth					= CL_DEVICE_IMAGE3D_MAX_DEPTH,
	image_max_buffer_size				= CL_DEVICE_IMAGE_MAX_BUFFER_SIZE,
	image_max_array_size				= CL_DEVICE_IMAGE_MAX_ARRAY_SIZE,
	max_samplers						= CL_DEVICE_MAX_SAMPLERS,
	max_parameter_size					= CL_DEVICE_MAX_PARAMETER_SIZE,
	mem_base_addr_align					= CL_DEVICE_MEM_BASE_ADDR_ALIGN,
	single_fp_config					= CL_DEVICE_SINGLE_FP_CONFIG,
	double_fp_config					= CL_DEVICE_DOUBLE_FP_CONFIG,
	global_mem_cache_type				= CL_DEVICE_GLOBAL_MEM_CACHE_TYPE,
	global_mem_cache_line_size			= CL_DEVICE_GLOBAL_MEM_CACHELINE_SIZE,
	global_mem_cache_size				= CL_DEVICE_GLOBAL_MEM_CACHE_SIZE,
	global_mem_size						= CL_DEVICE_GLOBAL_MEM_SIZE,
	max_constant_buffer_size			= CL_DEVICE_MAX_CONSTANT_BUFFER_SIZE,
	max_constant_args					= CL_DEVICE_MAX_CONSTANT_ARGS,
	local_mem_type						= CL_DEVICE_LOCAL_MEM_TYPE,
	local_mem_size						= CL_DEVICE_LOCAL_MEM_SIZE,
	error_correction_support			= CL_DEVICE_ERROR_CORRECTION_SUPPORT,
	host_unified_memory					= CL_DEVICE_HOST_UNIFIED_MEMORY,
	profiling_timer_resolution			= CL_DEVICE_PROFILING_TIMER_RESOLUTION,
	endian_little						= CL_DEVICE_ENDIAN_LITTLE,
	is_available						= CL_DEVICE_AVAILABLE,
	is_compiler_available				= CL_DEVICE_COMPILER_AVAILABLE,
	is_linker_available					= CL_DEVICE_LINKER_AVAILABLE,
	execution_capabilities				= CL_DEVICE_EXECUTION_CAPABILITIES,
	queue_properties					= CL_DEVICE_QUEUE_PROPERTIES,
	built_in_kernels					= CL_DEVICE_BUILT_IN_KERNELS,
	platform							= CL_DEVICE_PLATFORM,
	name								= CL_DEVICE_NAME,
	vendor								= CL_DEVICE_VENDOR,
	driver_version						= CL_DRIVER_VERSION,
	profile								= CL_DEVICE_PROFILE,
	device_version						= CL_DEVICE_VERSION,
	opencl_version						= CL_DEVICE_OPENCL_C_VERSION,
	extensions							= CL_DEVICE_EXTENSIONS,
	printf_buffer_size					= CL_DEVICE_PRINTF_BUFFER_SIZE,
	preferred_interop_user_sync			= CL_DEVICE_PREFERRED_INTEROP_USER_SYNC,
	parent_device						= CL_DEVICE_PARENT_DEVICE,
	partition_max_sub_devices			= CL_DEVICE_PARTITION_MAX_SUB_DEVICES,
	partition_properties				= CL_DEVICE_PARTITION_PROPERTIES,
	partition_affinity_domain			= CL_DEVICE_PARTITION_AFFINITY_DOMAIN,
	partition_type						= CL_DEVICE_PARTITION_TYPE,
	reference_count						= CL_DEVICE_REFERENCE_COUNT
};

enum class device_partition_property : cl_device_partition_property {
	unsupported						= 0,
	partition_equally				= CL_DEVICE_PARTITION_EQUALLY,
	partition_by_counts				= CL_DEVICE_PARTITION_BY_COUNTS,
	partition_by_affinity_domain	= CL_DEVICE_PARTITION_BY_AFFINITY_DOMAIN
};

enum class device_affinity_domain : cl_device_affinity_domain {
	unsupported			= 0,
	numa				= CL_DEVICE_AFFINITY_DOMAIN_NUMA,
	L4_cache			= CL_DEVICE_AFFINITY_DOMAIN_L4_CACHE,
	L3_cache			= CL_DEVICE_AFFINITY_DOMAIN_L3_CACHE,
	L2_cache			= CL_DEVICE_AFFINITY_DOMAIN_L2_CACHE,
	L1_cache			= CL_DEVICE_AFFINITY_DOMAIN_L1_CACHE,
	next_partitionable	= CL_DEVICE_AFFINITY_DOMAIN_NEXT_PARTITIONABLE
};

namespace detail {
using aff_domain_t = std::underlying_type<device_affinity_domain>::type;
}

enum class device_partition_type : detail::aff_domain_t {
	no_partition	= 0,
	numa			= (detail::aff_domain_t)device_affinity_domain::numa,
	L4_cache		= (detail::aff_domain_t)device_affinity_domain::L4_cache,
	L3_cache		= (detail::aff_domain_t)device_affinity_domain::L3_cache,
	L2_cache		= (detail::aff_domain_t)device_affinity_domain::L2_cache,
	L1_cache		= (detail::aff_domain_t)device_affinity_domain::L1_cache
};

enum class local_mem_type : cl_device_local_mem_type {
	none	= CL_NONE,
	local	= CL_LOCAL,
	global	= CL_GLOBAL
};

enum class fp_config : cl_device_fp_config {
	denorm							= CL_FP_DENORM,
	inf_nan							= CL_FP_INF_NAN,
	round_to_nearest				= CL_FP_ROUND_TO_NEAREST,
	round_to_zero					= CL_FP_ROUND_TO_ZERO,
	round_to_inf					= CL_FP_ROUND_TO_INF,
	fma								= CL_FP_FMA,
	correctly_rounded_divide_sqrt	= CL_FP_CORRECTLY_ROUNDED_DIVIDE_SQRT,
	soft_float						= CL_FP_SOFT_FLOAT
};

enum class global_mem_cache_type : cl_device_mem_cache_type {
	none		= CL_NONE,
	read_only	= CL_READ_ONLY_CACHE,
	write_only	= CL_READ_WRITE_CACHE
};

enum class device_execution_capabilities : cl_device_exec_capabilities {
	exec_kernel			= CL_EXEC_KERNEL,
	exec_native_kernel	= CL_EXEC_NATIVE_KERNEL
};


// C.4 Queue Information Descriptors
using queue_profiling = cl_command_queue_properties;
enum class queue : int {
	context,
	device,
	reference_count,
	properties
};


// C.7 Event Information Descriptors
enum class event : int {
	command_type,
	command_execution_status,
	reference_count
};

enum class event_profiling : int {
	command_queued,
	command_submit,
	command_start,
	command_end
};

} // namespace info
} // namespace sycl
} // namespace cl
