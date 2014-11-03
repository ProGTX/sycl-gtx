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
private:
	refc::ptr<cl_command_queue> command_q;
	context ctx;
	helper::error::handler handler;

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
