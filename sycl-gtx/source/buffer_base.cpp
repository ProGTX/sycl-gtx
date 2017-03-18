#include "SYCL/buffer_base.h"

#include "SYCL/queue.h"

using namespace cl::sycl;
using namespace detail;

::cl_int buffer_base::cl_enqueue_buffer(
    queue* q, ::size_t size, void* host_ptr,
    const vector_class<cl_event>& wait_events, cl_event& evnt,
    clEnqueueBuffer_f clEnqueueBuffer) {
  auto num_events_to_wait = wait_events.size();

  return clEnqueueBuffer(
      q->get(), device_data.get(), false,
      // TODO(progtx): Sub-buffer access
      0, size, host_ptr, static_cast<::cl_uint>(num_events_to_wait),
      (num_events_to_wait == 0 ? nullptr : wait_events.data()), &evnt);
}

cl_mem buffer_base::cl_create_buffer(queue* q, const cl_mem_flags& flags,
                                     ::size_t size, void* host_ptr,
                                     ::cl_int& error_code) {
  return clCreateBuffer(q->get_context().get(), flags, size, host_ptr,
                        &error_code);
}
