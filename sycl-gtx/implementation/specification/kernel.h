#pragma once

// 3.7 Expressing parallelism through kernels
// 3.7.1 is not included here, but rather in ranges.h

#include "../common.h"
#include "../debug.h"

namespace cl {
namespace sycl {

// Forward declarations
class program;
class context;

// 3.7.2.5 Kernel class

class kernel {
private:
	friend class program;
public:
	// The default object is not valid because there is no
	// program or cl_kernel associated with it
	kernel() = delete;

	// Constructs from a valid, initialized OpenCL kernel
	kernel(cl_kernel openclKernelObject);

	// Return the OpenCL kernel object for this kernel.
	cl_kernel get() const;

	// Return the context that this kernel is defined for.
	context get_context() const;

	// Return the program that this kernel is part of.
	program get_program() const;

	// Return the name of the kernel function.
	string_class get_kernel_attributes() const;

	template<cl_int name>
	typename detail::param_traits<cl_kernel_info, name>::param_type get_info() const;
};

// 3.7.3 Invoking kernels

// TODO: Passing kernel names
// Will need to divert slightly from the specification

template<class KernelType>
void single_task(string_class KernelName, KernelType kern) {
	using detail::cmd_group;
	cmd_group::check_scope();
	DSELF() << "not implemented.";
}

/*

template<typename KernelName, class KernelType>
void single_task(KernelType);

template<typename KernelName, class KernelType, int dimensions>
void parallel_for(range<dimensions> num_work_items, KernelType);

template<typename KernelName, class KernelType, int dimensions>
void parallel_for(range<dimensions> num_work_items, id<dimensions> work_item_offset, KernelType);

template<typename KernelName, class KernelType, int dimensions>
void parallel_for(nd_range<dimensions> execution_range, KernelType);

template<class KernelName, class WorkgroupFunctionType, int dimensions>
void parallel_for_work_group(range<dimensions> num_work_groups, WorkgroupFunctionType);

template<class KernelType, int dimensions>
void parallel_for_work_item(group num_work_items, KernelType);

*/

} // namespace sycl
} // namespace cl
