#include "context.h"
#include "device.h"
#include "platform.h"

using namespace cl::sycl;

detail::error::handler& context::default_error = detail::error::handler::default;

// Master constructor
// TODO: Deal with platform pointer
context::context(
	cl_context c,
	const cl_context_properties* properties,
	vector_class<device> target_devices_,
	const device_selector& dev_sel,
	detail::error::handler& handler_,
	platform* plt
) : ctx(c), handler(handler_), target_devices(target_devices_) {
	handler.set_thrower(this);
	if(c == nullptr) {
		cl_uint num_devices = target_devices.size();

		if(num_devices == 0) {
			target_devices = device_selector::get_platform().get_devices(dev_sel.type);
			num_devices = target_devices.size();
		}

		vector_class<cl_device_id> devices;
		devices.reserve(num_devices);
		for(auto& device_ptr : target_devices) {
			devices.push_back(device_ptr.get());
		}

		cl_int error_code;
		c = clCreateContext(properties, num_devices, devices.data(), nullptr, nullptr, &error_code);
		handler.report(error_code);
		ctx = c;
	}
	else {
		auto error_code = clRetainContext(c);
		handler.report(error_code);
	}
}

context::context()
	: context(nullptr, nullptr) {}
context::context(cl_context context)
	: context(context, nullptr) {}
context::context(const device_selector& deviceSelector, cl_context_properties* properties)
	: context(nullptr, properties, {}, deviceSelector) {}
context::context(const device& dev, cl_context_properties* properties)
	: context(nullptr, properties, {dev}) {}
context::context(const platform& plt, cl_context_properties* properties)
	: context(nullptr, properties, {}, *(device_selector::default)) {}
context::context(vector_class<device> deviceList, cl_context_properties* properties)
	: context(nullptr, properties, deviceList) {}

// TODO: Retain
cl_context context::get() const {
	return ctx.get();
}

// TODO:
vector_class<device> context::get_devices() const {
	return {};
}
