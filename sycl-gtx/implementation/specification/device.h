#pragma once

// Device classes

#include "refc.h"
#include "device_selector.h"
#include "error_handler.h"
#include "param_traits.h"
#include "../debug.h"
#include "../common.h"
#include <memory>

namespace cl {
namespace sycl {

// Forward declaration
class platform;

// 3.2.2 Device class
// Encapsulates a cl_device_id and a cl_platform_id
// In the case of constructing a device instance from an existing cl_device_id the system triggers a clRetainDevice.
// On destruction a call to clReleaseDevice is triggered.
class device {
private:
	// TODO: platform_id isn't set anywhere
	refc::ptr<cl_platform_id> platform_id;
	refc::ptr<cl_device_id> device_id;
	helper::error::handler handler;

	device(cl_device_id device_id, helper::error::handler handler);
public:
	device(cl_device_id device_id = nullptr, error_handler& handler = helper::error::handler::default);
	device(error_handler& handler);
	device(int& error_code);
	device(cl_device_id device_id, int& error_code);

	// Copy and move semantics
	device(const device&) = default;
#if MSVC_LOW
	device(device&& move)
		: SYCL_MOVE_INIT(platform_id), SYCL_MOVE_INIT(device_id), SYCL_MOVE_INIT(handler) {}
	friend void swap(device& first, device& second) {
		using std::swap;
		SYCL_SWAP(platform_id);
		SYCL_SWAP(device_id);
		SYCL_SWAP(handler);
	}
#else
	device(device&&) = default;
#endif

	cl_device_id get() const;

	// I believe there is an error in the specification and that this functions should be here instead of "platform get_platforms()"
	cl_platform_id get_platform() const;

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
		// Separate class implementation allows for partial template specialization
		return hidden<typename param_traits<cl_device_info, name>::param_type, name>::get_info(this);
	}
};

namespace helper {

VECTOR_CLASS<device> get_devices(
	cl_device_type device_type, refc::ptr<cl_platform_id> platform_id, error::handler& handler
);

unsigned int select_best_device(device_selector& selector, VECTOR_CLASS<device>& devices);

} // namespace helper

} // namespace sycl
} // namespace cl
