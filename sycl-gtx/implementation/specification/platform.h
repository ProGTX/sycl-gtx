#pragma once

// 3.2.1 Platform class

#include "device.h"
#include "platform.h"
#include "refc.h"
#include "../common.h"
#include "../error_handler.h"
#include <CL/cl.h>
#include <utility>
#include "../debug.h"

namespace cl {
namespace sycl {

class platform {
private:
	refc::ptr<cl_platform_id> platform_id;
	helper::err_handler handler;

	template<class Container, class Inner, size_t ArraySize>
	static VECTOR_CLASS<Container> to_vector(Inner(&array)[ArraySize], cl_uint size) {
		VECTOR_CLASS<Container> vector;
		vector.reserve(size);
		for(auto&& ptr : array) {
			vector.emplace_back(ptr);
		}
		return vector;
	}

public:
	platform(cl_platform_id platform_id, int& error_handler)
		: handler(error_handler), platform_id(refc::allocate(platform_id)) {}

	platform(cl_platform_id platform_id = nullptr)
		: platform_id(refc::allocate(platform_id)) {}

	platform(const platform&) = default;
	platform& operator=(const platform&) = default;

#if MSVC_LOW
	// Visual Studio [2013] does not support defaulted move constructors or move-assignment operators as the C++11 standard mandates.
	// http://msdn.microsoft.com/en-us/library/dn457344.aspx
	platform(platform&& move)
		: handler(std::move(move.handler)), platform_id(std::move(move.platform_id)) {}
	platform& operator=(platform&& move) {
		std::swap(platform_id, move.platform_id);
		std::swap(handler, move.handler);
		return *this;
	}
#else
	platform(platform&&) = default;
	platform& operator=(platform&&) = default;
#endif

	cl_platform_id get() const {
		return platform_id.get();
	}

	// Returns a vector of platforms.
	// Errors can be returned via C++ exceptions or via a reference to an error_code.
	static VECTOR_CLASS<platform> get_platforms(helper::err_handler::type& error_handler) {
		static const int MAX_PLATFORMS = 1024;
		cl_platform_id platforms_ids[MAX_PLATFORMS];
		cl_uint num_platforms;
		auto error_code = clGetPlatformIDs(MAX_PLATFORMS, platforms_ids, &num_platforms);
		helper::err_handler::handle(error_code, error_handler);
		return to_vector<platform>(platforms_ids, num_platforms);
	}
	static VECTOR_CLASS<platform> get_platforms() {
		return get_platforms(helper::err_handler::default_handler);
	}
	
	// TODO: There's probably an error in the specification - get_devices cannot be overloaded on "static" alone.

	// Returns a vector of corresponding devices.
	VECTOR_CLASS<device> get_devices(cl_device_type device_type = CL_DEVICE_TYPE_ALL) {
		static const int MAX_DEVICES = 1024;
		auto pid = platform_id.get();
		cl_device_id device_ids[MAX_DEVICES];
		cl_uint num_devices;
		auto error_code = clGetDeviceIDs(pid, device_type, MAX_DEVICES, device_ids, &num_devices);
		handler.handle(error_code);
		return to_vector<device>(device_ids, num_devices);
	}

	// TODO: Not yet sure what to do with param_types, it's not described in the specification
	// Direct equivalent of the OpenCL C API.
	//template<cl_int name>
	//typename param_traits<cl_platform_info, name>::param_type get_info();

	// TODO: How to check for this?
	bool is_host();

	bool has_extension(const STRING_CLASS extension_name) {
		// TODO: Maybe add caching
		static const int BUFFER_SIZE = 8192;
		char extensions[BUFFER_SIZE];
		auto pid = platform_id.get();
		auto error_code = clGetPlatformInfo(pid, CL_PLATFORM_EXTENSIONS, BUFFER_SIZE, extensions, NULL);
		handler.handle(error_code);

		STRING_CLASS ext_str(extensions);
		return ext_str.find(extension_name) != STRING_CLASS::npos;
	}
};

} // namespace sycl
} // namespace cl
