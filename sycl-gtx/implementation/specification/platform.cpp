#include "platform.h"
#include "device.h"

#include <utility>
#include "../debug.h"

using namespace cl::sycl;


platform::platform(cl_platform_id platform_id, device_selector& dev_selector)
	: platform_id(refc::allocate(platform_id)) {}
platform::platform()
	: platform(nullptr) {}
platform::platform(cl_platform_id platform_id)
	: platform(platform_id, *device_selector::default.get()) {}
platform::platform(device_selector &dev_selector)
	: platform(nullptr, dev_selector) {}

cl_platform_id platform::get() const {
	return platform_id.get();
}

vector_class<platform> platform::get_platforms() {
	static const int MAX_PLATFORMS = 1024;
	cl_platform_id platform_ids[MAX_PLATFORMS];
	cl_uint num_platforms;
	auto error_code = clGetPlatformIDs(MAX_PLATFORMS, platform_ids, &num_platforms);
	auto handler = detail::error::handler();
	handler.report(error_code);
	return vector_class<platform>(platform_ids, platform_ids + num_platforms);
}

vector_class<device> platform::get_devices(cl_device_type device_type) const {
	return detail::get_devices(device_type, platform_id.get(), handler);
}

// TODO: Check if SYCL running in Host Mode
bool platform::is_host() {
	DSELF() << "not implemented";
	return true;
}

bool platform::has_extension(const string_class extension_name) {
	return detail::has_extension<CL_PLATFORM_EXTENSIONS>(this, extension_name);
}
