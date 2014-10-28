#pragma once

// 3.4 Expressing parallelism through kernels
// 3.4.1 is not included here, but rather in ranges.h

#include "../common.h"
#include "../debug.h"
#include "program.h"

namespace cl {
namespace sycl {

// 3.4.2 Defining kernels

// typename T: kernel functor type
template <typename T>
class kernel {
	kernel<typename T>(context target_cont, device target_dev) {
		DSELF() << "not implemented";
	}
	kernel(context target_cont, device target_dev, STRING_CLASS string_kernel, STRING_CLASS string_name) {
		DSELF() << "not implemented";
	}

	cl_kernel get();
	context get_context();
	program get_program();
	STRING_CLASS get_kernel_attributes();
	STRING_CLASS get_function_name();
	void set_arg(int arg_index, accessor acc_obj);

	template<typename T>
	void set_arg(int arg_index, T scalar_value);
};

// 3.4.3 Invoking kernels

template <typename functorT>
void single_task(functorT f) {
	DSELF() << "not implemented";
}

template <typename functorT>
void parallel_for(int total_number_of_work_items, functorT f) {
	DSELF() << "not implemented";
}

template <typename functorT>
functorT kernel_lambda(STRING_CLASS name, functorT f) {
	DSELF() << "not implemented";
	return f;
}

template <typename functorT>
functorT kernel_functor(STRING_CLASS name, functorT f) {
	DSELF() << "not implemented";
	return f;
}

template <typename functorT>
functorT kernel_functor(functorT f) {
	DSELF() << "not implemented";
	return f;
}

} // namespace sycl
} // namespace cl
