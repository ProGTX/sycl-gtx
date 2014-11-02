#pragma once

// Device classes

#include "refc.h"
#include "error_handler.h"
#include "../debug.h"
#include "../common.h"
#include "../param_traits.h"

namespace cl {
namespace sycl {

// Forward declaration
class platform;

// 3.2.2 Device class
// Encapsulates a cl_device_id and a cl_platform_id
class device {
private:
	refc::ptr<cl_platform_id> platform_id;
	refc::ptr<cl_device_id> device_id;
	helper::error::handler handler;

public:
	// TODO: In the case of constructing a device instance from an existing cl_device_id the system triggers a clRetainDevice.
	device(cl_device_id device_id = nullptr, error_handler& handler = helper::error::handler::default);
	device(error_handler& handler);
	device(int& error_code);
	device(cl_device_id device_id, int& error_code);

	// TODO: On destruction a call to clReleaseDevice is triggered.
	~device() {}

	cl_device_id get() const;
	platform get_platforms();
	VECTOR_CLASS<device> get_devices(cl_device_type device_type = CL_DEVICE_TYPE_ALL);
	bool has_extension(const STRING_CLASS extension_name);
	bool is_host();
	bool is_cpu();
	bool is_gpu();

	// Partition device
	VECTOR_CLASS<device> create_sub_devices(
		const cl_device_partition_property* properties,
		int devices,
		unsigned int* num_devices
	);

private:
	template<class return_type, cl_int name>
	struct hidden {
		static return_type get_info(device* dev) {
			auto did = dev->device_id.get();
			return_type result;
			auto error_code = clGetDeviceInfo(did, name, sizeof(return_type), &result, nullptr);
			dev->handler.report(dev, error_code);
			return result;
		}
	};
	
public:
	template<cl_int name>
	typename param_traits<cl_device_info, name>::param_type get_info() {
		return hidden<typename param_traits<cl_device_info, name>::param_type, name>::get_info(this);
	}
};


namespace helper {

VECTOR_CLASS<device> get_devices(
	cl_device_type device_type, refc::ptr<cl_platform_id> platform_id, error::handler handler
);

} // namespace helper


// 3.2.4 Device selection class
// The class device_selector is an abstract class which enables the SYCL runtime to choose the best device based
// on heuristics specified by the user, or by one of the built-in device selectors
class device_selector {
public:
	device_selector() {
		DSELF() << "not implemented";
	}
	virtual int operator()(device dev) = 0;
};

// Built-in device selectors:
// Class name: gpu_selector
// Class name: cpu_selector
// Class name: host_selector

} // namespace sycl
} // namespace cl
