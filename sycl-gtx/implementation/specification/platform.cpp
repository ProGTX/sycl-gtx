#include "platform.h"
#include "device.h"

#include <utility>
#include "../debug.h"

using namespace cl::sycl;

platform::platform(cl_platform_id platform_id)
	: platform_id(refc::allocate(platform_id)) {}

cl_platform_id platform::get() const {
	return platform_id.get();
}

VECTOR_CLASS<platform> platform::get_platforms() {
	static const int MAX_PLATFORMS = 1024;
	cl_platform_id platforms_ids[MAX_PLATFORMS];
	cl_uint num_platforms;
	auto error_code = clGetPlatformIDs(MAX_PLATFORMS, platforms_ids, &num_platforms);
	return helper::to_vector<platform>(platforms_ids, num_platforms);
}

VECTOR_CLASS<device> platform::get_devices(cl_device_type device_type) {
	return helper::get_devices(device_type, platform_id);
}

// TODO: How to check for this?
bool platform::is_host() {
	DSELF() << "not implemented";
	return true;
}

bool platform::has_extension(const STRING_CLASS extension_name) {
	return helper::has_extension<CL_PLATFORM_EXTENSIONS>(this, extension_name);
}
