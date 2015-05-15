#pragma once

// 3.5.2 Platform class

#include "device_selector.h"
#include "refc.h"
#include "error_handler.h"
#include "param_traits.h"
#include "../common.h"

namespace cl {
namespace sycl {

// Forward declaration
class device;

// Constructors return errors via C++ exception class.
class platform {
private:
	detail::refc<cl_platform_id> platform_id;
	detail::error::handler handler;

	platform(cl_platform_id platform_id, const device_selector& dev_selector);

	static vector_class<platform> platforms;
public:
	// Default constructor for platform.
	// It constructs a platform object to encapsulate the device returned by the default device selector
	platform();

	// Construct a platform object from an OpenCL platform id.
	platform(cl_platform_id platform_id);

	// Construct a platform object from the device returned by a device selector of the user’s choice.
	platform(const device_selector& dev_selector);

	// Returns the cl platform id of the underlying OpenCL platform.
	// If the platform is not a valid OpenCL platform, for example it is the SYCL host,
	// a null cl platform id will be returned.
	cl_platform_id get() const;

	// Returns all available platforms in the system.
	static vector_class<platform> get_platforms();

	// Returns all the available devices of type device_type for this platform.
	vector_class<device> get_devices(cl_device_type device_type = CL_DEVICE_TYPE_ALL) const;

	// Direct equivalent of the OpenCL C API.
	// All parameters are char arrays, so the function is simplified
	template<cl_int name>
	typename string_class get_info() const {
		static const int BUFFER_SIZE = 8192;
		char buffer[BUFFER_SIZE];
		auto pid = platform_id.get();
		auto error_code = clGetPlatformInfo(pid, name, BUFFER_SIZE, buffer, nullptr);
		handler.report(error_code);
		return string_class(buffer);
	}

	bool is_host();
	bool has_extension(const string_class extension_name);
};

} // namespace sycl
} // namespace cl
