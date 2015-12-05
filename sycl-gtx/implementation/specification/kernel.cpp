#include "kernel.h"

#include "event.h"
#include "program.h"
#include "queue.h"

using namespace cl::sycl;

kernel::kernel(bool)
	:	prog(new program(ctx)) {}

kernel::kernel(cl_kernel k)
	:	kern(k),
		ctx(get_info<info::kernel::context>()),
		prog(new program(ctx, get_info<info::kernel::program>()))
{}

void kernel::enqueue_task(queue* q, const vector_class<cl_event>& wait_events, event* evnt) const {
	auto ev = evnt->evnt.get();

	auto error_code = clEnqueueTask(
		q->get(), kern.get(),
		wait_events.size(),
		get_events_ptr(wait_events),
		&ev
	);
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
