#pragma once

#include "event.h"
#include "../debug.h"
#include "../common.h"

namespace cl {
namespace sycl {

// Forward declaration
class queue;

namespace detail {

// Forward declaration
class issue_command;

class buffer_base {
protected:
	friend class issue_command;
	friend class queue;

	detail::refc<cl_mem, clRetainMemObject, clReleaseMemObject> device_data;
	vector_class<event> events;

	void create_accessor_command();

	using clEnqueueBuffer_f = decltype(&clEnqueueWriteBuffer);
	virtual void enqueue(queue* q, vector_class<cl_event>& wait_events, clEnqueueBuffer_f clEnqueueBuffer) {
		DSELF() << "not implemented";
	}
	static void enqueue_command(queue* q, vector_class<cl_event>& wait_events, buffer_base* buffer, clEnqueueBuffer_f clEnqueueBuffer) {
		buffer->enqueue(q, wait_events, clEnqueueBuffer);
	}
};

} // namespace detail

} // namespace sycl
} // namespace cl
