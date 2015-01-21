#include "invoke.h"

using namespace cl::sycl;

detail::kernel_::source* detail::kernel_::source::scope = nullptr;

void detail::kernel_::source::execute() {
	// TODO: Create kernel source
}
