#include "queue.h"

using namespace cl::sycl;

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

cl_command_queue queue::create_queue(info::queue_profiling properties) {
	display_device_info();

	cl_int error_code;
	auto q = clCreateCommandQueue(ctx.get(), dev.get(), properties, &error_code);
	detail::error::report(error_code);
	return q;
}


queue::queue(const async_handler& asyncHandler)
	: ctx(asyncHandler), command_q(create_queue()) {}

queue::queue(const device_selector& deviceSelector, const async_handler& asyncHandler)
	// TODO: Specification requires const selector in queue and non-const in device
	: dev(const_cast<device_selector&>(deviceSelector)), ctx(asyncHandler), command_q(create_queue()) {}

queue::queue(const context& syclContext, const device_selector& deviceSelector, const async_handler& asyncHandler)
	// TODO: Specification requires const selector in queue and non-const in device
	: dev(const_cast<device_selector&>(deviceSelector)), ctx(syclContext.get(), asyncHandler), command_q(create_queue()) {}

queue::queue(const context& syclContext, const device& syclDevice, const async_handler& asyncHandler)
	: queue(syclContext, syclDevice, false, asyncHandler) {}

// Chooses a device based on the provided device selector in the given context.
queue::queue(const context& syclContext, const device& syclDevice, info::queue_profiling profilingFlag, const async_handler& asyncHandler)
	: dev(syclDevice), ctx(syclContext.get(), asyncHandler), command_q(create_queue(profilingFlag)) {}

// Creates a queue for the provided device.
queue::queue(const device& syclDevice, const async_handler& asyncHandler)
	: dev(syclDevice), ctx(asyncHandler), command_q(create_queue()) {}

// Creates a SYCL queue from an OpenCL queue.
// At construction it does a retain on the queue memory object.
queue::queue(cl_command_queue clQueue, const async_handler& asyncHandler)
	: command_q(clQueue) {
	display_device_info();

	auto error_code = clRetainCommandQueue(clQueue);
	detail::error::report(error_code);

	ctx = context(get_info<CL_QUEUE_CONTEXT>(), asyncHandler);
	dev = device(get_info<CL_QUEUE_DEVICE>());
}

queue::~queue() {
	// TODO
	throw_asynchronous();
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
}

// TODO: This is a blocking wait for all enqueued tasks in the queue to complete.
void queue::wait() {
}

void queue::wait_and_throw() {
	wait();
	throw_asynchronous();
}
