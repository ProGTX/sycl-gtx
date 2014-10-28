#pragma once

// 3.2.5 Queue class

#include <CL/cl.h>
#include "../debug.h"
#include "context.h"
#include "device.h"

namespace cl {
namespace sycl {

// Encapsulation of an OpenCL cl_command_queue
class queue {
private:
	context contex;
public:
	queue() {
		DSELF() << "not implemented";
	}
	queue(cl_command_queue cmd_queue) {
		DSELF() << "not implemented";
	}
	queue(device_selector &selector) {
		DSELF() << "not implemented";
	}
	queue(context, device_selector &selector, cl_command_queue_properties = 0) {
		DSELF() << "not implemented";
	}
	queue(device queue_device) {
		DSELF() << "not implemented";
	}
	queue(queue &cmd_queue) {
		DSELF() << "not implemented";
	}

	// Optional parameters :
	//queue(..., error_handler &sync_handler);
	//queue(..., std::function &async_handler);
};


} // namespace sycl
} // namespace cl
