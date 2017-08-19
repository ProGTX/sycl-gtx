#include "SYCL/kernel.h"

#include "SYCL/event.h"
#include "SYCL/program.h"
#include "SYCL/queue.h"

using namespace cl::sycl;

kernel::kernel(bool) : prog(new program(ctx)) {}

kernel::kernel(cl_kernel k)
    : kern(k),
      ctx(get_info<info::kernel::context>()),
      prog(new program(ctx, get_info<info::kernel::program>())) {}

cl_event kernel::get_cl_event(event* evnt) {
  return evnt->evnt.get();
}
cl_command_queue kernel::get_cl_queue(queue* q) {
  return q->get();
}

void kernel::enqueue_task(queue* q, const vector_class<cl_event>& wait_events,
                          event* evnt) const {
  auto ev = evnt->evnt.get();

  auto error_code = clEnqueueTask(q->get(), kern.get(),
                                  static_cast<::cl_uint>(wait_events.size()),
                                  get_events_ptr(wait_events), &ev);
  detail::error::report(error_code);
}

program kernel::get_program() const {
  return *prog;
}

void kernel::set(cl_kernel openclKernelObject) {
  kern = openclKernelObject;
}

void kernel::set(const context& context, cl_program validProgram) {
  ctx = context;
  *prog = program(context, validProgram);
}
