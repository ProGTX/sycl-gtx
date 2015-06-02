#pragma once

// C. Interface of Memory Object Information Descriptors

#include <CL/cl.h>

namespace cl {
namespace sycl {
namespace info {

// C.1 Platform Information Descriptors
enum class platform : unsigned int {
	profile,
	version,
	name,
	vendor,
	extensions
};

// C.2 Context Information Descriptors
using gl_context_interop = cl_bool;
enum class context : int {
	reference_count,
	num_devices,
	devices,
	gl_interop
};

// TODO: C.3 Device Information Descriptors

enum class device_type : unsigned int {
	cpu			= CL_DEVICE_TYPE_CPU,
	gpu			= CL_DEVICE_TYPE_GPU,
	accelerator	= CL_DEVICE_TYPE_ACCELERATOR,
	custom		= CL_DEVICE_TYPE_CUSTOM,
	defaults	= CL_DEVICE_TYPE_DEFAULT,
	host,
	all			= CL_DEVICE_TYPE_ALL
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
