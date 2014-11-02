#include "context.h"
#include "device.h"

using namespace cl::sycl;

error_handler& context::default_error = helper::error::handler::default;

refc::ptr<cl_context> context::reserve(cl_context c) {
	return refc::allocate(c, clReleaseContext);
}

void context::construct(const cl_context_properties* properties, VECTOR_CLASS<device> target_devices) {
	auto c = ctx.get();
	if(c == nullptr) {
		cl_uint num_devices = target_devices.size();

		VECTOR_CLASS<cl_device_id> devices;
		devices.reserve(num_devices);
		for(auto& device_ptr : target_devices) {
			devices.push_back(device_ptr.get());
		}

		/* TODO
		void CL_CALLBACK (*pfn_notify)(
		const char *errinfo, const void* private_info, size_t cb, void* user_data
		);
		*/
		// TODO
		void* user_data = nullptr;

		cl_int error_code;
		c = clCreateContext(properties, num_devices, devices.data(), nullptr, user_data, &error_code);
		this->handler.report(this, error_code);
		ctx = reserve(c);
	}
	else {
		auto error_code = clRetainContext(c);
		this->handler.report(this, error_code);
	}
}

// Error handling via error_handler&
context::context(cl_context c, const cl_context_properties* properties, VECTOR_CLASS<device> target_devices, error_handler& handler)
	: ctx(reserve(c)), handler(handler) {
	construct(properties, target_devices);
}

// TODO: Auto load devices
context::context(cl_context c, error_handler& handler)
	: context(c, nullptr, {}, handler) {}

context::context(device_selector& dev_sel, error_handler& handler)
	: context(nullptr, dev_sel, handler) {}
context::context(const cl_context_properties* properties, device_selector& dev_sel, error_handler& handler)
	: ctx(reserve()), handler(handler) {
	// TODO: Selects best device
	device d;
	construct(properties, {d});
}

context::context(const cl_context_properties* properties, VECTOR_CLASS<device> target_devices, error_handler& handler)
	: context(nullptr, properties, target_devices, handler) {}
context::context(const cl_context_properties* properties, device target_device, error_handler& handler)
	: context(nullptr, properties, { target_device }, handler) {}

cl_context context::get() {
	return ctx.get();
}
