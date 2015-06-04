#pragma once

// C. Interface of Memory Object Information Descriptors

#include <CL/cl.h>

namespace cl {
namespace sycl {
namespace info {


// C.1 Platform Information Descriptors
enum class platform : unsigned int {
	profile		= CL_PLATFORM_PROFILE,
	version		= CL_PLATFORM_VERSION,
	name		= CL_PLATFORM_NAME,
	vendor		= CL_PLATFORM_VENDOR,
	extensions	= CL_PLATFORM_EXTENSIONS
};


// C.2 Context Information Descriptors
using gl_context_interop = cl_bool;
enum class context : unsigned int {
	reference_count	= CL_CONTEXT_REFERENCE_COUNT,
	num_devices		= CL_CONTEXT_NUM_DEVICES,
	devices			= CL_CONTEXT_DEVICES,
	gl_interop		= CL_CONTEXT_INTEROP_USER_SYNC	// TODO: Not sure
};


// C.3 Device Information Descriptors

using device_fp_config = unsigned int;
using device_exec_capabilities = unsigned int;
using device_queue_properties = unsigned int;

enum class device_type : unsigned int {
	cpu			= CL_DEVICE_TYPE_CPU,
	gpu			= CL_DEVICE_TYPE_GPU,
	accelerator	= CL_DEVICE_TYPE_ACCELERATOR,
	custom		= CL_DEVICE_TYPE_CUSTOM,
	defaults	= CL_DEVICE_TYPE_DEFAULT,
	host,
	all			= CL_DEVICE_TYPE_ALL
};

enum class device : unsigned int {
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

enum class device_partition_property : int {
	unsupported,
	partition_equally,
	partition_by_counts,
	partition_by_affinity_domain,
	partition_affinity_domain_next_partitionable
};

enum class device_affinity_domain : int {
	unsupported,
	numa,
	L4_cache,
	L3_cache,
	L2_cache,
	next_partitionable
};

enum class device_partition_type : int {
	no_partition,
	numa,
	L4_cache,
	L3_cache,
	L2_cache,
	L1_cache
};

enum class local_mem_type : int {
	none,
	local,
	global
};

enum class fp_config : int {
	denorm,
	inf_nan,
	round_to_nearest,
	round_to_zero,
	round_to_inf,
	fma,
	correctly_rounded_divide_sqrt,
	soft_float
};

enum class global_mem_cache_type : int {
	none,
	read_only,
	write_only
};

enum class device_execution_capabilities : unsigned int {
	exec_kernel,
	exec_native_kernel
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
