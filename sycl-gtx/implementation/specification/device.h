#pragma once

// 3.3.4 Device class

#include "refc.h"
#include "device_selector.h"
#include "error_handler.h"
#include "info.h"
#include "param_traits2.h"
#include "platform.h"
#include "ranges\id.h"
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
	template <class Contained_, info::device param, size_t BufferSize = detail::traits<Contained_>::BUFFER_SIZE>
	struct array_traits : detail::array_traits<Contained_, info::device, param, BufferSize> {
		void get_info(const device* dev) {
			Base::get(dev->device_id.get());
		}
	};

	template <class return_t, info::device param, class = typename std::is_enum<return_t>::type>
	struct traits;

	template <class return_t, info::device param>
	struct traits<return_t, param, typename std::enable_if<std::is_integral<return_t>::value, typename std::false_type::type>::type>
		: array_traits<return_t, param, 1> {
		return_t get(const device* dev) {
			get_info(dev);
			return param_value[0];
		}
	};

	template <class EnumClass, info::device param>
	struct traits<EnumClass, param, typename std::true_type::type> {
		EnumClass get(const device* dev) {
			return (EnumClass)traits<typename std::underlying_type<EnumClass>::type, param>().get(dev);
		}
	};

	template <typename EnumClass, info::device param>
	struct traits<vector_class<EnumClass>, param, typename std::true_type::type>
		: array_traits<typename std::underlying_type<EnumClass>::type, param> {
		using return_t = vector_class<EnumClass>;
		return_t convert() {
			return_t ret;
			auto size = actual_size / type_size;
			ret.reserve(size);
			for(size_t i = 0; i < size; ++i) {
				ret.push_back((EnumClass)param_value[i]);
			}
			return ret;
		}
		return_t get(const device* dev) {
			get_info(dev);
			return convert();
		}
	};

	template <info::device param>
	struct traits<string_class, param>
		: array_traits<string_class, param> {
		string_class get(const device* dev) {
			get_info(dev);
			return string_class(param_value);
		}
	};

	template <info::device param>
	struct traits<id<3>, param>
		: array_traits<size_t, param, 3> {
		id<3> get(const device* dev) {
			get_info(dev);
			return id<3>(param_value[0], param_value[1], param_value[2]);
		}
	};

	template <class Contained_>
	struct traits<vector_class<Contained_>, info::device::partition_type, typename std::false_type::type>
		: traits<vector_class<Contained_>, info::device::partition_type, typename std::true_type::type> {
		using return_t = vector_class<Contained_>;	// TODO: Why isn't return_t inherited? May be a bug.
		return_t get(const device* dev) {
			// TODO: I have no idea how to handle this case
			get_info(dev);
			if(actual_size == 0) {
				return_t ret;
				ret.push_back(info::device_partition_type::no_partition);
				return ret;
			}
			return convert();
		}
	};

public:
	template <info::device param>
	typename param_traits2<info::device, param>::type
	get_info() const {
		return traits<typename param_traits2<info::device, param>::type, param>().get(this);
	}
};

namespace detail {

vector_class<device> get_devices(
	cl_device_type device_type, cl_platform_id platform_id
);

} // namespace detail

} // namespace sycl
} // namespace cl
