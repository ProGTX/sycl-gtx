#pragma once

// 3.2.1 Platform class

#include <CL/cl.h>
#include <memory>
#include "../common.h"
#include "device.h"
#include "platform.h"

namespace cl {
namespace sycl {

class platform {
private:
	std::shared_ptr<cl_platform_id> platform_id;
public:
	platform();
	platform(cl_platform_id platform_id)
		: platform_id(platform_id) {}

#ifdef __CL_ENABLE_EXCEPTIONS
	platform(error_handler& handler);
	platform(cl_platform_id platform_id, error_handler& handler);
#else
	platform(int& error_code);
	platform(cl_platform_id platform_id, int& error_code);
#endif

	cl_platform_id get();
	static VECTOR_CLASS<platform> get_platforms();
	
	// TODO: Probably an error in the specification
	//VECTOR_CLASS<device> get_devices(cl_device_type device_type);

	static VECTOR_CLASS<device> get_devices(cl_device_type device_type = CL_DEVICE_TYPE_ALL);

	// TODO: Not yet sure what to do with param_types, it's not described in the specification
	//template<cl_int name>
	//typename param_traits<cl_platform_info, name>::param_type get_info();

	bool is_host();
	bool has_extension(const STRING_CLASS extension_name);
};

} // namespace sycl
} // namespace cl
