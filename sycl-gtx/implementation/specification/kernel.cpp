#include "kernel.h"

using namespace cl::sycl;

kernel::kernel(cl_kernel k)
	:	kern(refc::allocate<cl_kernel>(k, clReleaseKernel)),
		ctx(get_info<CL_KERNEL_CONTEXT>()),
		prog(ctx, get_info<CL_KERNEL_PROGRAM>())
{}
