#pragma once

// 3.3.5 Queue class

#include "context.h"
#include "device.h"
#include "error_handler.h"
#include "param_traits.h"
#include "refc.h"
#include "../common.h"
#include "../debug.h"

namespace cl {
namespace sycl {

// Encapsulation of an OpenCL cl_command_queue
class queue {
private:
	friend struct detail::error::report;

	static detail::error::handler& default_error;
	detail::error::handler handler = default_error;

	detail::refc<cl_command_queue, clRetainCommandQueue, clReleaseCommandQueue> command_q;
	device dev;
	context ctx;

	void display_device_info() const;
	void create_queue(cl_command_queue_properties* properties = nullptr);

public:
	// Creates a queue for a device it chooses according to the heuristics of the default selector.
	// The OpenCL context object is created implicitly.
	queue();

	// Creates a SYCL queue from an OpenCL queue.
	// At construction it does a retain on the queue memory object.
	// Returns errors via an exception.
	queue(cl_command_queue cl_queue);

	// Creates a queue for the device provided by the device selector.
	// If no device is selected, an error is reported via an exception.
	queue(const device_selector& selector);

	// Creates a queue for the provided device.
	// Any error is reported via an exception.
	queue(const device& queue_device);

	// Chooses a device based on the provided device selector in the given context.
	// If no device is selected, an error is reported via an exception.
	queue(const context& dev_context, device_selector& selector);

	// Creates a command queue using clCreateCommandQueue from a context and a device given the queue properties.
	// Returns errors via an exception.
	queue(const context& dev_context, const device& dev_device, cl_command_queue_properties* properties = nullptr);

	// TODO: queue(..., function_class<Args...>& async_handler);

	~queue();

	// Copy and move semantics
	queue(const queue&) = default;
#if MSVC_LOW
	queue(queue&& move)
		: SYCL_MOVE_INIT(command_q), SYCL_MOVE_INIT(ctx), SYCL_MOVE_INIT(dev), SYCL_MOVE_INIT(handler) {}
	friend void swap(queue& first, queue& second) {
		using std::swap;
		SYCL_SWAP(command_q);
		SYCL_SWAP(ctx);
		SYCL_SWAP(dev);
		SYCL_SWAP(handler);
	}
#else
	queue(queue&&) = default;
#endif

	// TODO: Returns the underlying OpenCL command queue after doing a retain.
	// Afterwards it needs to be manually released.
	cl_command_queue get();

	// Returns the SYCL context the queue is using.
	context get_context() const;

	// Returns the SYCL device the queue is associated with.
	device get_device() const;


	template<cl_command_queue_info name>
	using parameter_t = typename param_traits<cl_command_queue_info, name>::param_type;
	
	// Queries the platform for cl_command_queue info.
	template<cl_command_queue_info name>
	parameter_t<name> get_info() const {
		parameter_t<name> param_value;
		auto q = command_q.get();
		auto error_code = clGetCommandQueueInfo(q, name, sizeof(parameter_t<name>), &param_value, nullptr);
		handler.report(error_code);
		return param_value;
	}

	// Checks to see if any asynchronous errors have been produced by the queue
	// and if so reports them by passing them to the async_handler provided on construction.
	// If no async_handler was provided then asynchronous exceptions will be lost.
	void throw_asynchronous();

	// Performs a blocking wait for the completion all enqueued tasks in the queue.
	// Synchronous errors will be reported via an exception.
	void wait();

	// Performs a blocking wait for the completion of all enqueued tasks in the queue.
	void wait_and_throw();
};

} // namespace sycl
} // namespace cl
