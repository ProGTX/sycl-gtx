#include "context.h"
#include "device.h"

using namespace cl::sycl;

error_handler& context::default_error = helper::error::handler::default;

refc::ptr<cl_context> context::reserve(cl_context c) {
	return refc::allocate(c, clReleaseContext);
}

// TODO
device context::select_best_device(device_selector& dev_sel) {
	return device();
}

// Master constructor
context::context(
	cl_context c,
	const cl_context_properties* properties,
	VECTOR_CLASS<device> target_devices,
	error_handler& handler,
	context_notify* ctx_notify
) : ctx(reserve(c)), handler(handler) {
	if(c == nullptr) {
		cl_uint num_devices = target_devices.size();

		if(num_devices == 0) {
			// TODO: Auto load devices
		}

		VECTOR_CLASS<cl_device_id> devices;
		devices.reserve(num_devices);
		for(auto& device_ptr : target_devices) {
			devices.push_back(device_ptr.get());
		}

		auto pfn_notify = (ctx_notify == nullptr ? nullptr : &context_notify::forward);
		cl_int error_code;
		c = clCreateContext(properties, num_devices, devices.data(), pfn_notify, ctx_notify, &error_code);
		if(pfn_notify == nullptr) {
			this->handler.report(this, error_code);
		}
		ctx = reserve(c);
	}
	else {
		auto error_code = clRetainContext(c);
		this->handler.report(this, error_code);
	}
}

// Error handling via error_handler&
context::context(cl_context c, error_handler& handler)
	: context(c, nullptr, {}, handler) {}
context::context(device_selector& dev_sel, error_handler& handler)
	: context(nullptr, dev_sel, handler) {}
context::context(const cl_context_properties* properties, device_selector& dev_sel, error_handler& handler)
	: context(nullptr, properties, { select_best_device(dev_sel) }, handler, nullptr) {}
context::context(const cl_context_properties* properties, VECTOR_CLASS<device> target_devices, error_handler& handler)
	: context(nullptr, properties, target_devices, handler) {}
context::context(const cl_context_properties* properties, device target_device, error_handler& handler)
	: context(nullptr, properties, { target_device }, handler) {}

// Error handling via context_notify&
context::context(context_notify& handler)
	: context(nullptr, nullptr, {}, helper::error::handler::default, &handler) {}
context::context(cl_context c, context_notify& handler)
	: context(c, nullptr, {}, helper::error::handler::default, &handler) {}
context::context(device_selector& dev_sel, context_notify& handler)
	: context(nullptr, dev_sel, handler) {}
context::context(const cl_context_properties* properties, device_selector& dev_sel, context_notify& handler)
	: context(nullptr, properties, { select_best_device(dev_sel) }, helper::error::handler::default, &handler) {}
context::context(const cl_context_properties* properties, VECTOR_CLASS<device> target_devices, context_notify& handler)
	: context(nullptr, properties, target_devices, helper::error::handler::default, &handler) {}
context::context(const cl_context_properties* properties, device target_device, context_notify& handler)
	: context(nullptr, properties, { target_device }, helper::error::handler::default, &handler) {}

cl_context context::get() {
	return ctx.get();
}
