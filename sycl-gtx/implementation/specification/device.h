#pragma once

// 3.3.4 Device class

#include "refc.h"
#include "device_selector.h"
#include "error_handler.h"
#include "info.h"
#include "param_traits2.h"
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
	template <info::device_type type>
	bool is_type() const;

public:
	bool is_host() const;
	bool is_cpu() const;
	bool is_gpu() const;
	bool is_accelerator() const;

	platform get_platform() const;

	// Returns all the available OpenCL devices and the SYCL host device
	static vector_class<device> get_devices(info::device_type deviceType = info::device_type::all);

	bool has_extension(const string_class& extension_name);

	// Partition device
	vector_class<device> create_sub_devices(
		info::device_partition_type partitionType,
		info::device_partition_property partitionProperty,
		info::device_affinity_domain affinityDomain
	) const;

private:
	template <class return_t, info::device param>
	struct traits {
		static return_t get(const device* dev) {
			return_t param_value;
			detail::get_cl_info<info::device, param, sizeof(return_t)>(
				dev->device_id.get(), &param_value
			);
			return param_value;
		}
	};
	
	// TODO: Vectors of special types
	template <typename Contained, info::device param>
	struct traits<vector_class<Contained>, param> : detail::traits<Contained> {
		static return_t get(const device* dev) {
			Contained param_value[BUFFER_SIZE];
			size_t actual_size;
			detail::get_cl_info<info::device, param, BUFFER_SIZE * type_size>(
				dev->device_id.get(), param_value, &actual_size
			);
			return return_t(param_value, param_value + actual_size / type_size);
		}
	};

	template <info::device param>
	struct traits<string_class, param> : detail::traits<string_class>{
		static string_class get(const device* dev) {
			char param_value[BUFFER_SIZE];
			detail::get_cl_info<info::device, param, BUFFER_SIZE * type_size>(
				dev->device_id.get(), param_value
				);
			return string_class(param_value);
		}
	};

public:
	template <info::device param>
	typename param_traits2<info::device, param>::type
	get_info() const {
		return traits<typename param_traits2<info::device, param>::type, param>::get(this);
	}
};

namespace detail {

vector_class<device> get_devices(
	cl_device_type device_type, cl_platform_id platform_id
);

} // namespace detail

} // namespace sycl
} // namespace cl
