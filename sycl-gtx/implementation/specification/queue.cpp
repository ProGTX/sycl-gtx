#include "queue.h"

using namespace cl::sycl;

detail::error::handler& queue::default_error = detail::error::handler::default;

void queue::display_device_info() const {
	debug();
	debug() << "Queue device information:";
	debug() << dev.get_info<CL_DEVICE_NAME>();
	debug() << dev.get_info<CL_DEVICE_OPENCL_C_VERSION>();
	debug() << dev.get_info<CL_DEVICE_PROFILE>();
	debug() << dev.get_info<CL_DEVICE_VERSION>();
	debug() << dev.get_info<CL_DRIVER_VERSION>();
	debug();
}

void queue::create_queue(cl_command_queue_properties* properties) {
	display_device_info();

	cl_int error_code;
	command_q = clCreateCommandQueue(ctx.get(), dev.get(), ((properties == nullptr) ? 0 : *properties), &error_code);
	handler.report(error_code);
}

queue::queue()
	: queue(*(device_selector::default)) {}

queue::queue(cl_command_queue cl_queue)
	: handler(ctx), command_q(cl_queue) {
	display_device_info();

	auto error_code = clRetainCommandQueue(cl_queue);
	handler.report(error_code);

	ctx = context(get_info<CL_QUEUE_CONTEXT>());
	dev = device(get_info<CL_QUEUE_DEVICE>());
}

queue::queue(const device_selector& selector)
	// TODO: Specification requires const selector in queue and non-const in device
	: handler(ctx), dev(const_cast<device_selector&>(selector)), ctx(dev) {
	create_queue();
}

queue::queue(const device& queue_device)
	: handler(ctx), dev(queue_device), ctx(dev) {
	create_queue();
}

queue::queue(const context& dev_context, device_selector& selector)
	: handler(ctx), ctx(dev_context) {
	selector.select_device(ctx.get_devices());
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
