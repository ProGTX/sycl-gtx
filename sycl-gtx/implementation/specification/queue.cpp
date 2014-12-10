#include "queue.h"

using namespace cl::sycl;

error_handler& queue::default_error = detail::error::handler::default;

device queue::select_best_device(device_selector& selector, context& ctx) {
	auto device_pointers = ctx.get_info<CL_CONTEXT_DEVICES>();
	auto devices = detail::transform_vector<device>(device_pointers);
	auto index = detail::select_best_device(selector, devices);
	return devices[index];
}

context queue::create_context(queue* q, device_selector& selector) {
	q->ctx = context(selector);
	return std::move(q->ctx);
}

void queue::create_queue(cl_command_queue_properties* properties) {
	handler.set_thrower(&ctx);
	cl_int error_code;
	command_q = refc::allocate(
		clCreateCommandQueue(ctx.get(), dev.get(), ((properties == nullptr) ? 0 : *properties), &error_code),
		clReleaseCommandQueue
	);
	handler.report(error_code);
}

queue::queue()
	: ctx(), dev(select_best_device(*(device_selector::default), ctx)), handler(default_error) {
	handler.set_thrower(&ctx);
	cl_int error_code;
	command_q = refc::allocate(
		clCreateCommandQueue(ctx.get(), dev.get(), 0, &error_code),
		clReleaseCommandQueue
	);
	handler.report(error_code);
}

queue::queue(cl_command_queue cl_queue)
	: command_q(refc::allocate(cl_queue, clReleaseCommandQueue)), handler(default_error) {
	auto error_code = clRetainCommandQueue(cl_queue);
	handler.report(error_code);

	ctx = context(get_info<CL_QUEUE_CONTEXT>());
	handler.set_thrower(&ctx);

	dev = device(get_info<CL_QUEUE_DEVICE>());
}

queue::queue(const device_selector& selector) {}

queue::queue(const device& queue_device) {}

queue::queue(const context& dev_context, device_selector& selector) {}

queue::queue(const context& dev_context, const device& dev_device, cl_command_queue_properties* properties) {}

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
