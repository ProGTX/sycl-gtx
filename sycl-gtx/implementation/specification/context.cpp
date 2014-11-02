#include "context.h"
#include "device.h"

using namespace cl::sycl;

error_handler& context::default_error = helper::error::handler::default;

// Error handling via error_handler&
context::context(cl_context c, error_handler& handler)
	: ctx(refc::allocate<cl_context>(c, clReleaseContext)) {
	// TODO: In the case of copying the context it calls a clRetainContext
}
context::context(device_selector& dev_sel, error_handler& handler) {}
context::context(const cl_context_properties* properties, device_selector& dev_sel, error_handler& handler) {}
context::context(const cl_context_properties* properties, VECTOR_CLASS<device> target_devices, error_handler& handler) {}
context::context(const cl_context_properties* properties, device target_device, error_handler& handler) {}

cl_context context::get() {
	return ctx.get();
}
