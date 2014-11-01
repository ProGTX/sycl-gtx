#pragma once

// 3.2.1 Platform class

#include "refc.h"
#include "error_handler.h"
#include "../common.h"
#include "../param_traits.h"

namespace cl {
namespace sycl {

// Forward declaration
class device;

class platform {
private:
	refc::ptr<cl_platform_id> platform_id;

public:
	platform(cl_platform_id platform_id = nullptr);

	cl_platform_id get() const;

	// Returns a vector of platforms.
	// Errors can be returned via C++ exceptions or via a reference to an error_code.
	static VECTOR_CLASS<platform> get_platforms();
	
	// TODO: There's probably an error in the specification - get_devices cannot be overloaded on "static" alone.

	// Returns a vector of corresponding devices.
	VECTOR_CLASS<device> get_devices(cl_device_type device_type = CL_DEVICE_TYPE_ALL);

	// Direct equivalent of the OpenCL C API.
	template<cl_int name>
	typename param_traits<cl_platform_info, name>::param_type get_info() {
		static const int BUFFER_SIZE = 8192;
		char buffer[BUFFER_SIZE];
		auto pid = platform_id.get();
		auto error_code = clGetPlatformInfo(pid, name, BUFFER_SIZE, buffer, nullptr);
		return buffer;
	}

	bool is_host();
	bool has_extension(const STRING_CLASS extension_name);
};

} // namespace sycl
} // namespace cl
