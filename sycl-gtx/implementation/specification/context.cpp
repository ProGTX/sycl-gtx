#include "context.h"
#include "device.h"
#include "platform.h"

using namespace cl::sycl;

detail::error::handler& context::default_error = detail::error::handler::default;

refc::ptr<cl_context> context::reserve(cl_context c) {
	return refc::allocate(c, clReleaseContext);
}

vector_class<device> context::load_devices() {
	auto platforms = platform::get_platforms();
	// TODO: For now just select the first platform
	return platforms[0].get_devices();
}

// Master constructor
// TODO: Deal with platform pointer
context::context(
	cl_context c,
	const cl_context_properties* properties,
	vector_class<device> target_devices_,
	const device_selector& dev_sel,
	detail::error::handler& handler,
	platform* plt,
	context_notify* ctx_notify
) : ctx(reserve(c)), handler(handler), target_devices(target_devices_) {
	if(c == nullptr) {
		cl_uint num_devices = target_devices.size();

		if(num_devices == 0) {
			target_devices = load_devices();
			num_devices = target_devices.size();
		}
		best_device_id = detail::select_best_device(dev_sel, target_devices);
		if(best_device_id < 0) {
			// TODO: Maybe an exception?
		}

		vector_class<cl_device_id> devices;
		devices.reserve(num_devices);
		for(auto& device_ptr : target_devices) {
			devices.push_back(device_ptr.get());
		}

		auto pfn_notify = (ctx_notify == nullptr ? nullptr : &context_notify::forward);
		cl_int error_code;
		c = clCreateContext(properties, num_devices, devices.data(), pfn_notify, ctx_notify, &error_code);
		if(pfn_notify == nullptr) {
			this->handler.report(error_code);
		}
		ctx = reserve(c);
	}
	else {
		auto error_code = clRetainContext(c);
		this->handler.report(error_code);
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
