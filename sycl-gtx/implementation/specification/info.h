#pragma once

// C. Interface of Memory Object Information Descriptors

#include <CL/cl.h>

namespace cl {
namespace sycl {
namespace info {

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

} // namespace info
} // namespace sycl
} // namespace cl
