#include "queue.h"

using namespace cl::sycl;

queue::~queue() {
	throw_asynchronous();

	// TODO
}

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

// TODO: Which error?
cl_int queue::get_error() {
	return 0;
}

void queue::disable_exceptions() {
	exceptions_enabled = false;
}

// TODO: This function checks to see if any asynchronous errors have been thrown in the queue
// and if so reports them via exceptions, or via the supplied async_handler
void queue::throw_asynchronous() {
	handler.apply();
}

// TODO: This a blocking wait for all enqueued tasks in the queue to complete.
void queue::wait() {
}

void queue::wait_and_throw() {
	wait();
	throw_asynchronous();
}
