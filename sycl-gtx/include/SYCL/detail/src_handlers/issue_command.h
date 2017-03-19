#pragma once

#include "SYCL/detail/common.h"
#include "SYCL/detail/src_handlers/kernel_source.h"
#include "SYCL/kernel.h"

namespace cl {
namespace sycl {

// Forward declarations
class program;
class queue;

namespace detail {

class issue_command {
 private:
  static void compile_command(queue* q,
                              const vector_class<cl_event>& wait_events,
                              kernel_ns::source src,
                              shared_ptr_class<kernel> kern);
  static void prepare_kernel(shared_ptr_class<kernel> kern);

  static void enqueue_task_command(queue* q,
                                   const vector_class<cl_event>& wait_events,
                                   shared_ptr_class<kernel> kern, event* evnt);

  template <int dimensions>
  static void enqueue_range_command(queue* q,
                                    const vector_class<cl_event>& wait_events,
                                    shared_ptr_class<kernel> kern, event* evnt,
                                    range<dimensions> num_work_items,
                                    id<dimensions> offset) {
    prepare_kernel(kern);
    kern->enqueue_range(q, wait_events, evnt, num_work_items, offset);
  }

  template <int dimensions>
  static void enqueue_nd_range_command(
      queue* q, const vector_class<cl_event>& wait_events,
      shared_ptr_class<kernel> kern, event* evnt,
      nd_range<dimensions> execution_range) {
    prepare_kernel(kern);
    kern->enqueue_nd_range(q, wait_events, evnt, execution_range);
  }

 public:
  static void write_buffers_to_device(shared_ptr_class<kernel> kern);
  static void read_buffers_from_device(shared_ptr_class<kernel> kern);

  static void enqueue_task(shared_ptr_class<kernel> kern, event* evnt);

  template <int dimensions>
  static void enqueue_range(shared_ptr_class<kernel> kern, event* evnt,
                            range<dimensions> num_work_items,
                            id<dimensions> offset) {
    command::group_detail::add_kernel_enqueue_range(
        enqueue_range_command, __func__, kern, evnt, num_work_items, offset);
  }

  template <int dimensions>
  static void enqueue_nd_range(shared_ptr_class<kernel> kern, event* evnt,
                               nd_range<dimensions> execution_range) {
    command::group_detail::add_kernel_enqueue_nd_range(
        enqueue_nd_range_command, __func__, kern, evnt, execution_range);
  }
};

}  // namespace detail

}  // namespace sycl
}  // namespace cl
