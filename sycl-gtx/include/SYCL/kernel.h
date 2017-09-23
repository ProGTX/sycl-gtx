#pragma once

// 3.5.4 Kernel class

#include "SYCL/context.h"
#include "SYCL/detail/common.h"
#include "SYCL/detail/debug.h"
#include "SYCL/detail/src_handlers/kernel_source.h"
#include "SYCL/error_handler.h"
#include "SYCL/info.h"
#include "SYCL/param_traits.h"
#include "SYCL/ranges.h"
#include "SYCL/refc.h"
#include <algorithm>

namespace cl {
namespace sycl {

// Forward declarations
class context;
class event;
class queue;
class program;

class kernel {
 private:
  friend class program;
  friend class detail::issue_command;
  friend class detail::kernel_ns::source;

  detail::refc<cl_kernel, clRetainKernel, clReleaseKernel> kern;
  context ctx;
  shared_ptr_class<program> prog;
  detail::kernel_ns::source src;

  // These are meant only for program class
  kernel(bool);
  void set(cl_kernel openclKernelObject);
  void set(const context& context, cl_program validProgram);

 public:
  /**
   * The default object is not valid
   * because there is no program or cl_kernel associated with it
   */
  kernel() = delete;
  kernel(std::nullptr_t) = delete;

  /** Constructs from a valid, initialized OpenCL kernel */
  kernel(cl_kernel openclKernelObject);

  /** @return the OpenCL kernel object for this kernel. */
  cl_kernel get() const {
    return kern.get();
  }

  /** @return the context that this kernel is defined for. */
  context get_context() const {
    return ctx;
  }

  /** @return the program that this kernel is part of. */
  program get_program() const;

  template <info::kernel param>
  typename param_traits<info::kernel, param>::type get_info() const {
    using return_t = param_traits_t<info::kernel, param>;
    return detail::non_vector_traits<
               info::kernel, param,
               detail::traits_buffer_default<return_t>::size>()
        .get(kern.get());
  }

  /** @return the name of the kernel function */
  string_class get_kernel_attributes() const {
    return get_info<info::kernel::attributes>();
  }

  /** @return the name of the kernel function */
  string_class get_function_name() {
    return get_info<info::kernel::function_name>();
  }

 private:
  static cl_event get_cl_event(event* evnt);
  static cl_command_queue get_cl_queue(queue* q);

  static const cl_event* get_events_ptr(
      const vector_class<cl_event>& wait_events) {
    return (wait_events.size() == 0 ? nullptr : wait_events.data());
  }

  void enqueue_task(queue* q, const vector_class<cl_event>& wait_events,
                    event* evnt) const;

  template <int dimensions>
  void enqueue_range(queue* q, const vector_class<cl_event>& wait_events,
                     event* evnt, range<dimensions> num_work_items,
                     id<dimensions> offset) const {
    ::size_t* global_work_size = &num_work_items[0];
    ::size_t* offst = &static_cast<::size_t&>(offset[0]);
    auto ev = get_cl_event(evnt);

    auto error_code = clEnqueueNDRangeKernel(
        get_cl_queue(q), kern.get(), dimensions, offst, global_work_size,
        nullptr, static_cast<::cl_uint>(wait_events.size()),
        get_events_ptr(wait_events), &ev);
    detail::error::report(error_code);
  }

  template <int dimensions>
  void enqueue_nd_range(queue* q, const vector_class<cl_event>& wait_events,
                        event* evnt,
                        nd_range<dimensions> execution_range) const {
    ::size_t* local_work_size = &execution_range.get_local()[0];
    ::size_t* offst = &static_cast<::size_t&>(execution_range.get_offset()[0]);

    ::size_t global_work_size[dimensions];
    ::size_t* start = &execution_range.get_global()[0];
    std::copy(start, start + dimensions, global_work_size);

    // Adjust global work size
    for (int i = 0; i < dimensions; ++i) {
      auto remainder = global_work_size[i] % local_work_size[i];
      if (remainder > 0) {
        global_work_size[i] += local_work_size[i] - remainder;
      }
    }

    auto ev = get_cl_event(evnt);

    auto error_code = clEnqueueNDRangeKernel(
        get_cl_queue(q), kern.get(), dimensions, offst, global_work_size,
        local_work_size, static_cast<::cl_uint>(wait_events.size()),
        get_events_ptr(wait_events), &ev);
    detail::error::report(error_code);
  }
};

}  // namespace sycl
}  // namespace cl
