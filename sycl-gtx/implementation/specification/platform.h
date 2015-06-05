#pragma once

// 3.3.2 Platform class

#include "device_selector.h"
#include "error_handler.h"
#include "info.h"
#include "param_traits2.h"
#include "refc.h"
#include "../common.h"

namespace cl {
namespace sycl {

// Forward declaration
class device;

// Constructors return errors via C++ exception class.
class platform {
private:
	detail::refc<cl_platform_id> platform_id;

	platform(cl_platform_id platform_id, device_selector& dev_selector);

	static vector_class<platform> platforms;
public:
	// Default constructor for platform.
	// It constructs a platform object to encapsulate the device returned by the default device selector
	platform();

	// Construct a platform object from an OpenCL platform id.
	explicit platform(cl_platform_id platform_id);

	// Construct a platform object from the device returned by a device selector of the user’s choice.
	explicit platform(device_selector& dev_selector);

	// The OpenCL cl_platform_id or nullptr for SYCL host
	cl_platform_id get() const;

	// Returns all the available OpenCL platforms and the SYCL host platform
	static vector_class<platform> get_platforms();

	// Returns the devices available in this platform
	vector_class<device> get_devices(info::device_type = info::device_type::all) const;

private:
	template <info::platform param>
	struct string_traits : detail::array_traits<string_class, info::platform, param, 8192> {
		static string_class get(const platform* platfrm) {
			Base::get(platfrm->platform_id.get());
			return param_value;
		}
	};

public:
	// Returns the corresponding descriptor information for all SYCL platforms (OpenCL and host)
	template <info::platform param>
	typename param_traits2<info::platform, param>::type
	get_info() const {
		// Small optimization, knowing the return type is always string_class
		return string_traits<param>::get(this);
	}

	// True if the platform is host
	bool is_host() const;

	// Returns the available extensions for all SYCL platforms (OpenCL and host)
	bool has_extension(string_class extension_name) const;
};

} // namespace sycl
} // namespace cl
