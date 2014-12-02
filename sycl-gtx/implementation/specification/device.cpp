#include "device.h"
#include "platform.h"

using namespace cl::sycl;

device::device(cl_device_id device_id, detail::error::handler handler)
	: device_id(refc::allocate(device_id, clReleaseDevice)), handler(handler) {
	if(device_id != nullptr) {
		auto error_code = clRetainDevice(device_id);
		handler.report(error_code);
	}
	else {
		// TODO: The “default” device constructed corresponds to the host.
		// This is also the device that the system will “fall-back” to,
		// if there are no existing or valid OpenCL devices associated with the system.
		debug::warning(__func__) << "does not support a default device yet";
	}
}

device::device(cl_device_id device_id, error_handler& handler)
	: device(device_id, detail::error::handler(handler)) {}

device::device(error_handler& handler)
	: device(nullptr, handler) {}

device::device(int& error_code)
	: device(nullptr, error_code) {}

device::device(cl_device_id device_id, int& error_code)
	: device(device_id, detail::error::handler(error_code)) {}

cl_device_id device::get() const {
	return device_id.get();
}

cl_platform_id device::get_platform() const {
	return platform_id.get();
}

vector_class<device> device::get_devices(cl_device_type device_type) {
	return detail::get_devices(device_type, platform_id, handler);
}

bool device::has_extension(const string_class extension_name) {
	return detail::has_extension<CL_DEVICE_EXTENSIONS>(this, extension_name);
}

vector_class<device> device::create_sub_devices(
	const cl_device_partition_property* properties,
	int devices,
	unsigned int* num_devices
) {
	auto did = device_id.get();
	cl_device_id* device_ids = new cl_device_id[devices];
	auto error_code = clCreateSubDevices(did, properties, devices, device_ids, num_devices);
	handler.report(error_code);
	auto device_vector = vector_class<device>(device_ids, device_ids + *num_devices);
	delete[] device_ids;
	return device_vector;
}

vector_class<device> detail::get_devices(
	cl_device_type device_type, refc::ptr<cl_platform_id> platform_id, error::handler& handler
) {
	static const int MAX_DEVICES = 1024;
	auto pid = platform_id.get();
	cl_device_id device_ids[MAX_DEVICES];
	cl_uint num_devices;
	auto error_code = clGetDeviceIDs(pid, device_type, MAX_DEVICES, device_ids, &num_devices);
	handler.report(error_code);
	return vector_class<device>(device_ids, device_ids + num_devices);
}

unsigned int detail::select_best_device(device_selector& selector, vector_class<device>& devices) {
	unsigned int best_id = -1;
	int best_score = -1;
	int i = 0;

	for(auto&& dev : devices) {
		int score = selector(dev);
		if(score > best_score) {
			best_id = i;
			best_score = score;
		}
		++i;
	}

	// Devices with a negative score will never be chosen.
	return (best_score >= 0 ? best_id : -1);
}
