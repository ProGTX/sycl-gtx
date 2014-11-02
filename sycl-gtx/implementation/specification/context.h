#pragma once

#include "refc.h"
#include "error_handler.h"
#include "../common.h"
#include "../debug.h"
#include "../param_traits.h"

namespace cl {
namespace sycl {

// Forward declarations
class device;
class device_selector;
class program;

// Used as the notification function for contexts.
class context_notify {
public:
	virtual void operator()(const STRING_CLASS errinfo, const void* private_info, size_t cb) = 0;
	virtual void operator()(program t_program) = 0;
};

class context {
private:
	refc::ptr<cl_context> ctx;
	helper::error::handler handler;
	static error_handler& default_error;

public:
	// TODO: The constructor creates a context and in the case of copying it calls a clRetainContext

	// TODO: Error handling via error_handler&
	context(cl_context c = nullptr, error_handler& handler = default_error);
	context(device_selector& dev_sel, error_handler& handler = default_error);
	context(const cl_context_properties* properties, device_selector& dev_sel, error_handler& handler = default_error);
	context(const cl_context_properties* properties, VECTOR_CLASS<device> target_devices, error_handler& handler = default_error);
	context(const cl_context_properties* properties, device target_device, error_handler& handler = default_error);

	// TODO: Error handling via context_notify&
	context(context_notify& handler);
	context(cl_context c, context_notify& handler);
	context(device_selector& dev_sel, context_notify& handler);
	context(const cl_context_properties* properties, device_selector& dev_sel, context_notify& handler);
	context(const cl_context_properties* properties, VECTOR_CLASS<device> target_devices, context_notify& handler);
	context(const cl_context_properties* properties, device target_device, context_notify& handler);

	~context() {}

	cl_context get();

	// TODO: Deal with array types
	template<cl_int name>
	typename param_traits<cl_context_info, name>::param_type get_info() {
		auto c = ctx.get();
		param_traits<cl_context_info, name>::param_type param_value;
		size_t param_value_size = sizeof(decltype(param_value));
		auto error_code = clGetContextInfo(c, name, param_value_size, &param_value, nullptr);
		handler.handle(error_code);
		return param_value;
	}
};

} // namespace sycl
} // namespace cl
