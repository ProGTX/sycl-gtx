#pragma once

// 3.2.5 Queue class

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
public:
	using async_handler_t = helper::error::async_handler::function_t;

private:
	refc::ptr<cl_command_queue> command_q;
	context ctx;
	device dev;

	helper::error::handler handler;
	static error_handler& default_error;
	bool exceptions_enabled = true;

	static device select_best_device(device_selector& selector, context& ctx);

	// Master constructor
	queue(context ctx, device dev, cl_command_queue_properties properties, error_handler& sync_handler, bool host_fallback);
public:
	// Create commmand queue from existing one
	queue(cl_command_queue cmd_queue, error_handler& sync_handler = default_error);

	// Creates a command queue using clCreateCommandQueue from a context and a device.
	// Returns errors via C++ exceptions.
	queue(context ctx, device dev, cl_command_queue_properties properties = 0, error_handler& sync_handler = default_error);

	// This chooses a device to run the command_groups on based on the provided selector.
	// If no device is selected, runs on the host.
	// If no selector is provided, the method for choosing the "best" device is undefined.
	// This constructor cannot report an error, as any error during queue creation enforces queue creation on the host.
	queue(device_selector& selector = *device_selector::default, cl_command_queue_properties properties = 0, error_handler& sync_handler = default_error);

	// This chooses a device to run the command_groups based on the provided selector, but must be within the provided context.
	// If no device is selected, an error is reported via a C++ exception.
	queue(context ctx, device_selector& selector, cl_command_queue_properties properties = 0, error_handler& sync_handler = default_error);

	// This creates a queue on the given device.
	// Any error is reported via C++ exceptions.
	queue(device dev, cl_command_queue_properties properties = 0, error_handler& sync_handler = default_error);

	//TODO: queue(..., std::function &async_handler);

	~queue();

	queue(const queue& cmd_queue) = default;
	queue& operator=(const queue& cmd_queue) = default;

	cl_command_queue get();
	context get_context();
	device get_device();
	cl_int get_error();
	
	template<cl_int name>
	typename param_traits<cl_command_queue_info, name>::param_type get_info() {
		using type = param_traits<cl_command_queue_info, name>::param_type;
		type param_value;
		auto q = command_q.get();
		auto error_code = clGetCommandQueueInfo(q, name, sizeof(type), &param_value, nullptr);
		handler.report(this, error_code);
		return param_value;
	}

	void disable_exceptions();
	void throw_asynchronous();
	void wait();
	void wait_and_throw();
};

} // namespace sycl
} // namespace cl
