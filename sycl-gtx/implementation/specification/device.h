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
	// TODO: platform_id isn't set anywhere, first we need to select a platform
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

	// TODO
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
		using real_return = return_type;
		static real_return get_info(device* dev) {
			auto did = dev->device_id.get();
			real_return param_value;
			auto error_code = clGetDeviceInfo(did, name, sizeof(real_return), &param_value, nullptr);
			dev->handler.report(dev, error_code);
			return param_value;
		}
	};
	template<class return_type, cl_int name>
	struct hidden<return_type[], name> {
		using real_return = VECTOR_CLASS<return_type>;
		static real_return get_info(device* dev) {
			auto did = dev->device_id.get();
			static const int BUFFER_SIZE = 1024;
			return_type param_value[BUFFER_SIZE];
			std::size_t actual_size;
			auto error_code = clGetDeviceInfo(did, name, BUFFER_SIZE * sizeof(return_type), param_value, &actual_size);
			dev->handler.report(dev, error_code);
			return real_return(param_value, param_value + actual_size);
		}
	};
	template<cl_int name>
	struct hidden<char[], name> {
		using real_return = STRING_CLASS;
		static real_return get_info(device* dev) {
			auto did = dev->device_id.get();
			static const int BUFFER_SIZE = 8192;
			char param_value[BUFFER_SIZE];
			auto error_code = clGetDeviceInfo(did, name, BUFFER_SIZE * sizeof(char), param_value, nullptr);
			dev->handler.report(dev, error_code);
			return real_return(param_value);
		}
	};
	template<cl_int name>
	using param = typename param_traits<cl_device_info, name>::param_type;
public:
	template<cl_int name>
	typename hidden<param<name>, name>::real_return get_info() {
		return hidden<param<name>, name>::get_info(this);
	}

	void a() {
		auto b1 = get_info<CL_DEVICE_ADDRESS_BITS>();
		auto b2 = get_info<CL_DEVICE_AVAILABLE>();
		auto b3 = get_info<CL_DEVICE_DOUBLE_FP_CONFIG>();
		auto b4 = get_info<CL_DEVICE_PARTITION_TYPE>();
		auto b5 = get_info<CL_DEVICE_NAME>();
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
