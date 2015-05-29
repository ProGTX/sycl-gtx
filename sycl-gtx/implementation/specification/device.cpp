#include "device.h"
#include "platform.h"

using namespace cl::sycl;

device::device(cl_device_id device_id, device_selector* dev_sel)
	: device_id(device_id), platfrm(*dev_sel) {
	if(device_id != nullptr) {
		auto error_code = clRetainDevice(device_id);
		handler.report(error_code);
	}
	else {
		*this = dev_sel->select_device();
	}
}

device::device()
	: device(nullptr, device_selector::default.get()) {}

device::device(cl_device_id device_id)
	: device(device_id, nullptr) {}

device::device(device_selector& dev_sel)
	: device(nullptr, &dev_sel) {}

cl_device_id device::get() const {
	return device_id.get();
}

cl_platform_id device::get_platform() const {
	return platfrm.get();
}

vector_class<device> device::get_devices(cl_device_type device_type) {
	return detail::get_devices(device_type, platfrm.get(), handler);
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
	cl_device_type device_type, cl_platform_id platform_id, const error::handler& handler
) {
	static const int MAX_DEVICES = 1024;
	cl_device_id device_ids[MAX_DEVICES];
	cl_uint num_devices;
	auto error_code = clGetDeviceIDs(platform_id, device_type, MAX_DEVICES, device_ids, &num_devices);
	handler.report(error_code);
	return vector_class<device>(device_ids, device_ids + num_devices);
}
