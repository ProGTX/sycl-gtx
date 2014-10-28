#pragma once

// 3.4 Expressing parallelism through kernels
// 3.4.1 is not included here, but rather in ranges.h

#include "../common.h"

namespace cl {
namespace sycl {

// 3.4.2 Defining kernels

// typename T: kernel functor type
template <typename T>
class kernel {
	kernel<typename T>(context target cont, device target dev) {}
	kernel(context target_cont, device target_dev, STRING_CLASS string_kernel, STRING_CLASS string_name) {}

	cl_kernel get();
	context get_context();
	program get_program();
	STRING_CLASS get_kernel_attributes();
	STRING_CLASS get_function_name();
	void set_arg(int arg index, accessor acc obj);

	template<typename T>
	void set_arg(int arg index, T scalar value);
};

// 3.4.3 Invoking kernels

} // namespace sycl
} // namespace cl
