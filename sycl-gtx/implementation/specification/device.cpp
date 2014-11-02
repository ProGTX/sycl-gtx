#include "device.h"
#include "platform.h"

using namespace cl::sycl;

device::device(cl_device_id device_id, helper::error::handler handler)
	: device_id(refc::allocate(device_id, clReleaseDevice)), handler(handler) {
	if(device_id != nullptr) {
		auto error_code = clRetainDevice(device_id);
		handler.report(this, error_code);
	}
	else {
		// TODO: The “default” device constructed corresponds to the host.
		// This is also the device that the system will “fall-back” to,
		// if there are no existing or valid OpenCL devices associated with the system.
		debug::warning(__func__) << "does not support a default device yet";
	}
}

device::device(cl_device_id device_id, error_handler& handler)
	: device(device_id, helper::error::handler(handler)) {}

device::device(error_handler& handler)
	: device(nullptr, handler) {}

device::device(int& error_code)
	: device(nullptr, error_code) {}

device::device(cl_device_id device_id, int& error_code)
	: device(device_id, helper::error::handler(error_code)) {}

cl_device_id device::get() const {
	return device_id.get();
}

cl_platform_id device::get_platform() const {
	return platform_id.get();
}

VECTOR_CLASS<device> device::get_devices(cl_device_type device_type) {
	return helper::get_devices(device_type, platform_id, handler);
}

bool device::has_extension(const STRING_CLASS extension_name) {
	return helper::has_extension<CL_DEVICE_EXTENSIONS>(this, extension_name);
}

VECTOR_CLASS<device> device::create_sub_devices(
	const cl_device_partition_property* properties,
	int devices,
	unsigned int* num_devices
) {
	auto did = device_id.get();
	cl_device_id* device_ids = new cl_device_id[devices];
	auto error_code = clCreateSubDevices(did, properties, devices, device_ids, num_devices);
	handler.report(this, error_code);
	auto device_vector = helper::to_vector<device>(device_ids, *num_devices, true);
	delete[] device_ids;
	return device_vector;
}

VECTOR_CLASS<device> helper::get_devices(
	cl_device_type device_type, refc::ptr<cl_platform_id> platform_id, error::handler& handler
) {
	static const int MAX_DEVICES = 1024;
	auto pid = platform_id.get();
	cl_device_id device_ids[MAX_DEVICES];
	cl_uint num_devices;
	auto error_code = clGetDeviceIDs(pid, device_type, MAX_DEVICES, device_ids, &num_devices);
	handler.report(device_ids, error_code);
	return to_vector<device>(device_ids, num_devices);
}
