#include "device.h"

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
