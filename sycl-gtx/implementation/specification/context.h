#pragma once

#include "refc.h"
#include "../common.h"
#include "../debug.h"

namespace cl {
namespace sycl {

// Forward declarations
class device;
class device_selector;
class program;

class context {
private:
	refc::ptr<cl_context> ctx;

public:
	// TODO: The constructor creates a context and in the case of copying it calls a clRetainContext
	context() {
		DSELF() << "not implemented";
	}
	context(device_selector dev_sel);
	context(const cl_context_properties* properties, device_selector dev_sel);
	context(const cl_context_properties* properties, VECTOR_CLASS<device> target_devices);
	context(const cl_context_properties* properties, device target_device);
	context(cl_context c);

	// TODO: On destruction clReleaseContext is called.
	~context() {}

	cl_context get();

	//template<cl_int name>
	//typename param_traits<cl_context_info, name>::param_type get_info();
};

// Used as the notification function for contexts.
class context_notify {
public:
	context_notify();
	~context_notify();
	virtual void operator()(const STRING_CLASS errinfo, const void* private_info, size_t cb) = 0;
	virtual void operator()(program t_program) = 0;
};

} // namespace sycl
} // namespace cl
