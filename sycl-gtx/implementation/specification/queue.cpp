#include "queue.h"

using namespace cl::sycl;

cl_command_queue queue::get() {
	return command_q.get();
}

context queue::get_context() {
	return ctx;
}

// TODO: Which device?
device queue::get_device() {
	return device();
}
