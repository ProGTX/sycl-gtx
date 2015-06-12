#include "handler.h"
#include "kernel.h"
#include "queue.h"

using namespace cl::sycl;

kernel::kernel(cl_kernel k)
	:	kern(k),
		ctx(get_info<info::kernel::context>()),
		prog(ctx, get_info<info::kernel::program>()),
		cg([](handler&) {})
{}

kernel::~kernel() {
	// Prevent custom commands from executing
	cg.commands.clear();
}

void kernel::enqueue_task(queue* q) const {
	auto error_code = clEnqueueTask(
		q->get(), kern.get(),
		// TODO: Events
		0, nullptr, nullptr
	);
	detail::error::report(error_code);
}
