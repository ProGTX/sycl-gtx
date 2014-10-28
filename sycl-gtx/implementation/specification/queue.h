#pragma once

// 3.2.5 Queue class

#include <CL/cl.h>
#include "context.h"
#include "device_selector.h"

namespace cl {
namespace sycl {

// Encapsulation of an OpenCL cl_command_queue
class queue {
public:
	queue() {}
	queue(cl_command_queue cmd_queue) {}
	queue(device_selector &selector) {}
	queue(context, device_selector &selector, cl_command_queue_properties = 0) {}
	queue(device queue_device) {}
	queue(queue &cmd_queue) {}

	// Optional parameters :
	//queue(..., error_handler &sync_handler);
	//queue(..., std::function &async_handler);
};


} // namespace sycl
} // namespace cl
