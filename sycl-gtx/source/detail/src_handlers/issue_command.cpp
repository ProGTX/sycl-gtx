#include "SYCL/detail/src_handlers/issue_command.h"

#include "SYCL/accessors/buffer.h"
#include "SYCL/buffer.h"
#include "SYCL/kernel.h"

using namespace cl::sycl;
using detail::issue_command;
using namespace detail::kernel_ns;

// TODO(progtx):
void issue_command::compile_command(queue* q,
                                    const vector_class<cl_event>& wait_events,
                                    source src, shared_ptr_class<kernel> kern) {
}

void issue_command::prepare_kernel(shared_ptr_class<kernel> kern) {
  DSELF() << kern->src.kernel_name;
  auto k = kern->get();
  ::cl_int error_code;
  int i = 0;
  for (auto& acc : kern->src.resources) {
    if (acc.second.acc.target == access::target::local) {
      error_code = clSetKernelArg(k, i, acc.second.size, nullptr);
    } else {
      auto mem = acc.second.acc.data->device_data.get();
      error_code = clSetKernelArg(k, i, acc.second.size, &mem);
    }
    detail::error::report(error_code);
    ++i;
  }
}

void issue_command::write_buffers_to_device(shared_ptr_class<kernel> kern) {
  for (auto& acc : kern->src.resources) {
    auto mode = acc.second.acc.mode;
    if (mode == access::mode::write || mode == access::mode::discard_write ||
        mode == access::mode::discard_read_write ||
        acc.second.acc.target == access::target::local) {
      // Don't need to copy data that won't be used
      continue;
    }
    command::group_detail::add_buffer_copy(
        acc.second.acc, access::mode::write, buffer_base::enqueue_command,
        __func__, acc.second.acc.data, &clEnqueueWriteBuffer);
  }
}

void issue_command::enqueue_task_command(
    queue* q, const vector_class<cl_event>& wait_events,
    shared_ptr_class<kernel> kern, event* evnt) {
  prepare_kernel(kern);
  kern->enqueue_task(q, wait_events, evnt);
}

void issue_command::enqueue_task(shared_ptr_class<kernel> kern, event* evnt) {
  command::group_detail::add_kernel_enqueue_task(enqueue_task_command, __func__,
                                                 kern, evnt);
}

void issue_command::read_buffers_from_device(shared_ptr_class<kernel> kern) {
  for (auto& acc : kern->src.resources) {
    if (acc.second.acc.mode == access::mode::read ||
        acc.second.acc.target == access::target::local) {
      // Don't need to read back read-only buffers
      continue;
    }
    command::group_detail::add_buffer_copy(
        acc.second.acc, access::mode::read, buffer_base::enqueue_command,
        __func__, acc.second.acc.data,
        reinterpret_cast<buffer_base::clEnqueueBuffer_f>(  // NOLINT
            &clEnqueueReadBuffer));
  }
}
