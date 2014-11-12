#pragma once

// 3.2.1 Platform class

#include "refc.h"
#include "error_handler.h"
#include "param_traits.h"
#include "../common.h"

namespace cl {
namespace sycl {

// Forward declaration
class device;

class platform {
private:
	refc::ptr<cl_platform_id> platform_id;
	detail::error::handler handler;

public:
	platform(cl_platform_id platform_id = nullptr, error_handler& handler = detail::error::handler::default);
	platform(error_handler& handler);
	platform(int& error_code);
	platform(cl_platform_id platform_id, int& error_code);

	cl_platform_id get() const;

	// Returns a vector of platforms.
	// Errors can be returned via C++ exceptions or via a reference to an error_code.
	static VECTOR_CLASS<platform> get_platforms(error_handler& handler = detail::error::handler::default);
	static VECTOR_CLASS<platform> get_platforms(int& error_code);
private:
	static VECTOR_CLASS<platform> get_platforms(detail::error::handler& handler);

public:
	// TODO: There's probably an error in the specification - get_devices cannot be overloaded on "static" alone.

	// Returns a vector of corresponding devices.
	VECTOR_CLASS<device> get_devices(cl_device_type device_type = CL_DEVICE_TYPE_ALL);

	// Direct equivalent of the OpenCL C API.
	// All parameters are char arrays, so the function is simplified
	template<cl_int name>
	typename STRING_CLASS get_info() {
		static const int BUFFER_SIZE = 8192;
		char buffer[BUFFER_SIZE];
		auto pid = platform_id.get();
		auto error_code = clGetPlatformInfo(pid, name, BUFFER_SIZE, buffer, nullptr);
		handler.report(this, error_code);
		return STRING_CLASS(buffer);
	}

	bool is_host();
	bool has_extension(const STRING_CLASS extension_name);
};

} // namespace sycl
} // namespace cl
