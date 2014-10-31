#include "device.h"
#include "platform.h"


VECTOR_CLASS<cl::sycl::device> cl::sycl::helper::get_devices(
	cl_device_type device_type, refc::ptr<cl_platform_id> platform_id, err_handler handler
) {
	static const int MAX_DEVICES = 1024;
	auto pid = platform_id.get();
	cl_device_id device_ids[MAX_DEVICES];
	cl_uint num_devices;
	auto error_code = clGetDeviceIDs(pid, device_type, MAX_DEVICES, device_ids, &num_devices);
	handler.handle(error_code);
	return to_vector<device>(device_ids, num_devices);
}

using namespace cl::sycl;

device::device(cl_device_id device_id)
	: device_id(refc::allocate(device_id)) {}

device::device(cl_device_id device_id, int& error_handler)
	: handler(error_handler), device_id(refc::allocate(device_id)) {}

#if MSVC_LOW
device::device(device&& move)
	: handler(std::move(move.handler)), device_id(std::move(move.device_id)) {}
device& device::operator=(device&& move) {
	std::swap(platform_id, move.platform_id);
	std::swap(device_id, move.device_id);
	std::swap(handler, move.handler);
	return *this;
}
#endif

cl_device_id device::get() const {
	return device_id.get();
}

// TODO: Plural name, but only returns one platform?
platform device::get_platforms() {
	return platform(platform_id.get());
}

VECTOR_CLASS<device> device::get_devices(cl_device_type device_type) {
	return helper::get_devices(device_type, platform_id, handler);
}

bool device::has_extension(const STRING_CLASS extension_name) {
	return helper::has_extension<CL_DEVICE_EXTENSIONS>(this, extension_name);
}
