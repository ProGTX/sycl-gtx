#include "queue.h"

using namespace cl::sycl;

detail::error::handler& queue::default_error = detail::error::handler::default;

void queue::create_queue(cl_command_queue_properties* properties) {
	cl_int error_code;
	command_q = refc::allocate(
		clCreateCommandQueue(ctx.get(), dev.get(), ((properties == nullptr) ? 0 : *properties), &error_code),
		clReleaseCommandQueue
	);
	handler.report(error_code);
}

queue::queue()
	: queue(*(device_selector::default)) {}

queue::queue(cl_command_queue cl_queue)
	: handler(ctx), command_q(refc::allocate(cl_queue, clReleaseCommandQueue)) {
	auto error_code = clRetainCommandQueue(cl_queue);
	handler.report(error_code);

	ctx = context(get_info<CL_QUEUE_CONTEXT>());
	dev = device(get_info<CL_QUEUE_DEVICE>());
}

queue::queue(const device_selector& selector)
	: handler(ctx), dev(selector), ctx(dev) {
	create_queue();
}

queue::queue(const device& queue_device)
	: handler(ctx), dev(queue_device), ctx(dev) {
	create_queue();
}

queue::queue(const context& dev_context, device_selector& selector)
	: handler(ctx), ctx(dev_context) {
	auto devices = ctx.get_devices();
	auto best_id = detail::best_device_id(selector, devices);
	if(best_id < 0) {
		// TODO: Report no device selected.
		//handler.report();
	}
	else {
		dev = std::move(devices[best_id]);
	}
	create_queue();
}

queue::queue(const context& dev_context, const device& dev_device, cl_command_queue_properties* properties)
	: handler(ctx), dev(dev_device), ctx(dev_context) {
	create_queue(properties);
}

queue::~queue() {
	throw_asynchronous();

	// TODO
}

cl_command_queue queue::get() {
	return command_q.get();
}

context queue::get_context() const {
	return ctx;
}

device queue::get_device() const {
	return dev;
}

// TODO: This function checks to see if any asynchronous errors have been thrown in the queue
// and if so reports them via exceptions, or via the supplied async_handler
void queue::throw_asynchronous() {
	handler.apply();
}

// TODO: This is a blocking wait for all enqueued tasks in the queue to complete.
void queue::wait() {
}

void queue::wait_and_throw() {
	wait();
	throw_asynchronous();
}
