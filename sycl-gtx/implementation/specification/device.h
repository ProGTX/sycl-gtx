#pragma once

// 3.3.4 Device class

#include "refc.h"
#include "device_selector.h"
#include "error_handler.h"
#include "info.h"
#include "param_traits.h"
#include "platform.h"
#include "../debug.h"
#include "../common.h"

namespace cl {
namespace sycl {

// Encapsulates a particular SYCL device against on which kernels may be executed
class device {
private:
	detail::refc<cl_device_id, clRetainDevice, clReleaseDevice> device_id;
	platform platfrm;

	device(cl_device_id device_id, device_selector* selector);
public:
	// Default constructor for the device.
	// It choses a device using default selector.
	device();

	// Constructs a device class instance using cl device_id of the OpenCL device.
	explicit device(cl_device_id device_id);

	// Constructs a device class instance using the device selector provided.
	explicit device(device_selector& selector);

	// Copy and move semantics
	device(const device&) = default;
#if MSVC_LOW
	device(device&& move)
		: SYCL_MOVE_INIT(device_id), SYCL_MOVE_INIT(platfrm) {}
	friend void swap(device& first, device& second) {
		using std::swap;
		SYCL_SWAP(device_id);
		SYCL_SWAP(platfrm);
	}
#else
	device(device&&) = default;
#endif

	cl_device_id get() const;

private:
	bool is_type(info::device_type type) const;

public:
	bool is_host() const;
	bool is_cpu() const;
	bool is_gpu() const;
	bool is_accelerator() const;

	platform get_platform() const;

	// Returns all the available OpenCL devices and the SYCL host device
	static vector_class<device> get_devices(info::device_type deviceType = info::device_type::all);

	bool has_extension(const string_class extension_name);

	// Partition device
	vector_class<device> create_sub_devices(
		info::device_partition_type partitionType,
		info::device_partition_property partitionProperty,
		info::device_affinity_domain affinityDomain
	) const;

private:
	template<class return_type, cl_int name>
	struct hidden {
		using real_return = return_type;
		static real_return get_info(const device* dev) {
			auto did = dev->device_id.get();
			real_return param_value;
			auto error_code = clGetDeviceInfo(did, name, sizeof(real_return), &param_value, nullptr);
			detail::error::report(error_code);
			return param_value;
		}
	};
	template<class return_type, cl_int name>
	struct hidden<return_type[], name> {
		using real_return = vector_class<return_type>;
		static real_return get_info(const device* dev) {
			auto did = dev->device_id.get();
			static const int BUFFER_SIZE = 1024;
			return_type param_value[BUFFER_SIZE];
			std::size_t actual_size;
			std::size_t type_size = sizeof(return_type);
			auto error_code = clGetDeviceInfo(did, name, BUFFER_SIZE * type_size, param_value, &actual_size);
			detail::error::report(error_code);
			return real_return(param_value, param_value + actual_size / type_size);
		}
	};
	template<cl_int name>
	struct hidden<char[], name> {
		using real_return = string_class;
		static real_return get_info(const device* dev) {
			auto did = dev->device_id.get();
			static const int BUFFER_SIZE = 8192;
			char param_value[BUFFER_SIZE];
			auto error_code = clGetDeviceInfo(did, name, BUFFER_SIZE * sizeof(char), param_value, nullptr);
			detail::error::report(error_code);
			return real_return(param_value);
		}
	};
	template<cl_int name>
	using param = typename param_traits<cl_device_info, name>::param_type;
public:
	template<cl_int name>
	typename hidden<param<name>, name>::real_return get_info() const {
		return hidden<param<name>, name>::get_info(this);
	}
};

namespace detail {

vector_class<device> get_devices(
	cl_device_type device_type, cl_platform_id platform_id
);

} // namespace detail

} // namespace sycl
} // namespace cl
