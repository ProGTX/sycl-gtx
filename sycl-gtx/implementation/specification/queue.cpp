#include "queue.h"

#include "buffer_base.h"

using namespace cl::sycl;

void queue::display_device_info() const {
	debug();
	debug() << "Queue device information:";
	debug() << dev.get_info<info::device::name>();
	debug() << dev.get_info<info::device::opencl_version>();
	debug() << dev.get_info<info::device::profile>();
	debug() << dev.get_info<info::device::device_version>();
	debug() << dev.get_info<info::device::driver_version>();
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
	: ctx(asyncHandler), command_q(create_queue()), command_group(this) {}

queue::queue(const device_selector& deviceSelector, const async_handler& asyncHandler)
	// TODO: Specification requires const selector in queue and non-const in device
	: dev(const_cast<device_selector&>(deviceSelector)), ctx(asyncHandler), command_q(create_queue()), command_group(this) {}

queue::queue(const context& syclContext, const device_selector& deviceSelector, const async_handler& asyncHandler)
	// TODO: Specification requires const selector in queue and non-const in device
	: dev(const_cast<device_selector&>(deviceSelector)), ctx(syclContext.get(), asyncHandler), command_q(create_queue()), command_group(this) {}

queue::queue(const context& syclContext, const device& syclDevice, const async_handler& asyncHandler)
	: queue(syclContext, syclDevice, false, asyncHandler) {}

// Chooses a device based on the provided device selector in the given context.
queue::queue(const context& syclContext, const device& syclDevice, info::queue_profiling profilingFlag, const async_handler& asyncHandler)
	: dev(syclDevice), ctx(syclContext.get(), asyncHandler), command_q(create_queue(profilingFlag)), command_group(this) {}

// Creates a queue for the provided device.
queue::queue(const device& syclDevice, const async_handler& asyncHandler)
	: dev(syclDevice), ctx(asyncHandler), command_q(create_queue()), command_group(this) {}

// Creates a SYCL queue from an OpenCL queue.
// At construction it does a retain on the queue memory object.
queue::queue(cl_command_queue clQueue, const async_handler& asyncHandler)
	: command_q(clQueue), command_group(this) {
	display_device_info();

	ctx = context(get_info<info::queue::context>(), asyncHandler);
	dev = device(get_info<info::queue::device>());
}

queue::~queue() {
	wait_and_throw();
}

bool queue::is_host() {
	return dev.is_host();
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

// Checks to see if any asynchronous errors have been produced by the queue
// and if so reports them by passing them to the async_handler passed to the queue on construction.
// If no async_handler was provided then asynchronous exceptions will be lost.
void queue::throw_asynchronous() {
	if(ex_list.size() > 0) {
		detail::error::thrower::report_async(&ctx, ex_list);
	}
}

void queue::wait() {
	auto error_code = clFinish(command_q.get());
	detail::error::report(error_code);
}

void queue::wait_and_throw() {
	wait();
	throw_asynchronous();
}

vector_class<cl_event> queue::get_wait_events(const std::set<detail::buffer_base*>& dependencies) const {
	vector_class<cl_event> wait_events;

	for(auto&& buf : dependencies) {
		if(buffers_in_use.count(buf) > 0) {
			wait_events.reserve(wait_events.size() + buf->events.size());
			for(auto& ev : buf->events) {
				wait_events.push_back(ev.get());
			}
		}
	}

	return wait_events;
}
