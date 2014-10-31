#include "platform.h"
#include "device.h"

#include <utility>
#include "../debug.h"

using namespace cl::sycl;

platform::platform(cl_platform_id platform_id, int& error_handler)
	: handler(error_handler), platform_id(refc::allocate(platform_id)) {}

platform::platform(cl_platform_id platform_id)
	: platform_id(refc::allocate(platform_id)) {}

#if MSVC_LOW
platform::platform(platform&& move)
	: handler(std::move(move.handler)), platform_id(std::move(move.platform_id)) {}
platform& platform::operator=(platform&& move) {
	std::swap(platform_id, move.platform_id);
	std::swap(handler, move.handler);
	return *this;
}
#endif

cl_platform_id platform::get() const {
	return platform_id.get();
}

VECTOR_CLASS<platform> platform::get_platforms(helper::err_handler::type& error_handler) {
	static const int MAX_PLATFORMS = 1024;
	cl_platform_id platforms_ids[MAX_PLATFORMS];
	cl_uint num_platforms;
	auto error_code = clGetPlatformIDs(MAX_PLATFORMS, platforms_ids, &num_platforms);
	helper::err_handler::handle(error_code, error_handler);
	return helper::to_vector<platform>(platforms_ids, num_platforms);
}

VECTOR_CLASS<platform> platform::get_platforms() {
	return get_platforms(helper::err_handler::default_handler);
}

VECTOR_CLASS<device> platform::get_devices(cl_device_type device_type) {
	return helper::get_devices(device_type, platform_id, handler);
}

// TODO: How to check for this?
bool platform::is_host() {
	DSELF() << "not implemented";
	return true;
}

bool platform::has_extension(const STRING_CLASS extension_name) {
	return helper::has_extension(this, extension_name);
}
