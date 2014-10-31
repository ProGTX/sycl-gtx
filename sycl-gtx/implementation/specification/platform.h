#pragma once

// 3.2.1 Platform class

#include "refc.h"
#include "../common.h"
#include "../error_handler.h"
#include "../param_traits.h"

namespace cl {
namespace sycl {

// Forward declaration
class device;

class platform {
private:
	refc::ptr<cl_platform_id> platform_id;
	helper::err_handler handler;

public:
	platform(cl_platform_id platform_id, int& error_handler);
	platform(cl_platform_id platform_id = nullptr);
	platform(const platform&) = default;
	platform& operator=(const platform&) = default;

#if MSVC_LOW
	// Visual Studio [2013] does not support defaulted move constructors or move-assignment operators as the C++11 standard mandates.
	// http://msdn.microsoft.com/en-us/library/dn457344.aspx
	platform(platform&& move);
	platform& operator=(platform&& move);
#else
	platform(platform&&) = default;
	platform& operator=(platform&&) = default;
#endif

	cl_platform_id get() const;

	// Returns a vector of platforms.
	// Errors can be returned via C++ exceptions or via a reference to an error_code.
	static VECTOR_CLASS<platform> get_platforms();
	static VECTOR_CLASS<platform> get_platforms(helper::err_handler::type& error_handler);
	
	// TODO: There's probably an error in the specification - get_devices cannot be overloaded on "static" alone.

	// Returns a vector of corresponding devices.
	VECTOR_CLASS<device> get_devices(cl_device_type device_type = CL_DEVICE_TYPE_ALL);

	// Direct equivalent of the OpenCL C API.
	template<cl_int name>
	typename param_traits<cl_platform_info, name>::param_type get_info() {
		static const int MAX_BUFFER = 8192;
		auto pid = platform_id.get();
		size_t param_value_size;
		char buffer[MAX_BUFFER];
		size_t param_value_size_ret;
		auto error_code = clGetPlatformInfo(pid, name, param_value_size, buffer, &param_value_size_ret);
		handler.handle(error_code);
		return buffer;
	}

	bool is_host();
	bool has_extension(const STRING_CLASS extension_name);
};

} // namespace sycl
} // namespace cl
