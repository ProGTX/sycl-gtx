#pragma once

#include "kernel.h"
#include "ranges.h"
#include "../common.h"
#include "../gen_source.h"

// 3.7.3 Invoking kernels

namespace cl {
namespace sycl {

// TODO: Passing kernel names
// From my understanding of the specification (revision 2014-09-16, section 2.6),
// the kernel name isn't really needed here - can generate own name

// 3.7.3.1 Single Task invoke
template<class KernelType>
void single_task(KernelType kernFunctor) {
	detail::command::group_::check_scope();
	auto src = detail::kernel_::constructor<void>::get(kernFunctor);
	auto kern = src.compile();
	debug() << "Compiled kernel:";
	debug() << src.get_code();
	src.write_buffers_to_device();
	src.enqueue_task(kern);
	src.read_buffers_from_device();
}


// 3.7.3.2 Parallel For invoke

template<class KernelType, int dimensions>
void parallel_for(range<dimensions> num_work_items, KernelType kernFunctor) {
	detail::command::group_::check_scope();
	auto src = detail::kernel_::constructor<item<dimensions>>::get(kernFunctor, num_work_items);
	auto kern = src.compile();
	debug() << "Compiled kernel:";
	debug() << src.get_code();
	src.write_buffers_to_device();
	src.enqueue_range(kern, num_work_items);
	src.read_buffers_from_device();
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
