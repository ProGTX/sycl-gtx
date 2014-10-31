#pragma once

// Device classes

#include "refc.h"
#include "../debug.h"
#include "../common.h"
#include "../error_handler.h"
#include "../param_traits.h"

namespace cl {
namespace sycl {

// Forward declaration
class platform;

// 3.2.2 Device class
// Encapsulates a cl_device_id and a cl_platform_id
class device {
private:
	refc::ptr<cl_platform_id> platform_id;
	refc::ptr<cl_device_id> device_id;
	helper::err_handler handler;

public:
	device(cl_device_id device_id = nullptr);
	device(cl_device_id device_id, int& error_handler);
	device(const device&) = default;
	device& operator=(const device&) = default;

#if MSVC_LOW
	// Visual Studio [2013] does not support defaulted move constructors or move-assignment operators as the C++11 standard mandates.
	// http://msdn.microsoft.com/en-us/library/dn457344.aspx
	device(device&& move);
	device& operator=(device&& move);
#else
	device(device&&) = default;
	device& operator=(device&&) = default;
#endif

	cl_device_id get() const;
	platform get_platforms();
	VECTOR_CLASS<device> get_devices(cl_device_type device_type = CL_DEVICE_TYPE_ALL);
	
	template<cl_int name>
	typename param_traits<cl_device_info, name>::param_type get_info() {
		return param_traits<cl_device_info, name>::param_type();
	}
	
	bool has_extension(const STRING_CLASS extension_name);
	bool is_host();
	bool is_cpu();
	bool is_gpu();
	VECTOR_CLASS<device> create_sub_devices(
		const cl_device_partition_property* properties,
		int devices,
		unsigned int* num_devices
	);
};


namespace helper {

VECTOR_CLASS<device> get_devices(
	cl_device_type device_type,
	refc::ptr<cl_platform_id> platform_id,
	err_handler handler
);

} // namespace helper


// 3.2.4 Device selection class
// The class device_selector is an abstract class which enables the SYCL runtime to choose the best device based
// on heuristics specified by the user, or by one of the built-in device selectors
class device_selector {
public:
	device_selector() {
		DSELF() << "not implemented";
	}
	virtual int operator()(device dev) = 0;
};

// Built-in device selectors:
// Class name: gpu_selector
// Class name: cpu_selector
// Class name: host_selector

} // namespace sycl
} // namespace cl
