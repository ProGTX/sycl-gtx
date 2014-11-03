#pragma once

// 3.2.5 Queue class

#include "../common.h"
#include "../debug.h"
#include "context.h"
#include "device.h"
#include "error_handler.h"
#include "refc.h"

namespace cl {
namespace sycl {

// Encapsulation of an OpenCL cl_command_queue
class queue {
private:
	refc::ptr<cl_command_queue> command_q;
	context ctx;

	static device select_best_device(device_selector& selector);

public:
	queue(cl_command_queue cmd_queue = nullptr, error_handler& sync_handler = helper::error::handler::default) {}
	queue(device_selector& selector, error_handler& sync_handler = helper::error::handler::default);
	queue(context ctx, device_selector& selector, cl_command_queue_properties properties = 0, error_handler& sync_handler = helper::error::handler::default);
	queue(device queue_device, error_handler& sync_handler = helper::error::handler::default);

	//TODO: queue(..., std::function &async_handler);

	queue(const queue& cmd_queue) = default;
	queue& operator=(const queue& cmd_queue) = default;

	cl_command_queue get();
	context get_context();
	device get_device();
	cl_int get_error();
	template<cl_int name>
	typename param_traits<cl_command_queue_info, name>::param_type get_info();
	void disable_exceptions();
	void throw_asynchronous();
	void wait();
	void wait_and_throw();
};

} // namespace sycl
} // namespace cl
