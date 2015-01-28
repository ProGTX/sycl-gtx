#pragma once

#include "../common.h"

// 3.7.3 Invoking kernels

namespace cl {
namespace sycl {

// TODO: Passing kernel names
// Will need to divert slightly from the specification
// Diversion could be avoided if I could get functor name at compile time

template<class KernelType>
void single_task(string_class kernelName, KernelType kern) {
	detail::cmd_group::check_scope();
	auto src = detail::kernel_::source::generate(kernelName, kern);
	debug() << src;
	// TODO: Enqueue kernel invocation
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
