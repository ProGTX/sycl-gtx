#include "kernel.h"

#include "context.h"

using namespace cl::sycl;

kernel::kernel(cl_kernel)
	: prog(context(get_info<CL_KERNEL_CONTEXT>()), get_info<CL_KERNEL_PROGRAM>()) {
}
