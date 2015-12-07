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
	friend class ::cl::sycl::queue;

	detail::refc<cl_mem, clRetainMemObject, clReleaseMemObject> device_data;
	vector_class<event> events;

	void create_accessor_command();

	using clEnqueueBuffer_f = decltype(&clEnqueueWriteBuffer);
	virtual void enqueue(queue* q, const vector_class<cl_event>& wait_events, clEnqueueBuffer_f clEnqueueBuffer) {
		DSELF() << "not implemented";
	}
	static void enqueue_command(
		queue* q, const vector_class<cl_event>& wait_events, buffer_base* buffer, clEnqueueBuffer_f clEnqueueBuffer
	) {
		buffer->enqueue(q, wait_events, clEnqueueBuffer);
	}
	cl_int cl_enqueue_buffer(
		queue* q,
		size_t size,
		void* host_ptr,
		const vector_class<cl_event>& wait_events,
		cl_event& evnt,
		clEnqueueBuffer_f clEnqueueBuffer
	);

	static cl_mem cl_create_buffer(queue* q, const cl_mem_flags& flags, size_t size, void* host_ptr, cl_int& error_code);
};

} // namespace detail

} // namespace sycl
} // namespace cl
