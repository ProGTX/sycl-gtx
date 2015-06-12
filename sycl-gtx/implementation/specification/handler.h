#pragma once

// 3.5.3 SYCL functions for invoking kernels

#include "access.h"
#include "kernel.h"
#include "ranges.h"
#include "../common.h"
#include "../function_traits.h"
#include "../gen_source.h"
#include "../invoke_source.h"

namespace cl {
namespace sycl {

// 3.5.3.4 Command group handler class
class handler {
private:
	friend class command_group;
	// TODO: Implementation defined constructor
	handler() {}

public:
	template <typename DataType, int dimensions, access::mode mode, access::target target>
	void set_arg(int arg_index, accessor<DataType, dimensions, mode, target>& acc_obj);

	template <typename T>
	void set_arg(int arg_index, T scalar_value);

	// TODO: Passing kernel names
	// From my understanding of the specification I can generate my own kernel name

	// 3.5.3.1 Single Task invoke
	template <class KernelType>
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


	// 3.5.3.2 Parallel For invoke

	template <class KernelType, int dimensions>
	void parallel_for(range<dimensions> numWorkItems, KernelType kernFunctor) {
		parallel_for(numWorkItems, id<dimensions>(), kernFunctor);
	}

	// This type of kernel can be invoked with a function accepting either an id or an item as parameter
	template <class KernelType, int dimensions>
	void parallel_for(range<dimensions> numWorkItems, id<dimensions> workItemOffset, KernelType kernFunctor) {
		detail::command::group_::check_scope();
		auto src = detail::kernel_::constructor<typename detail::first_arg<KernelType>::type>::get(
			kernFunctor, numWorkItems, workItemOffset
		);
		auto kern = src.compile();
		debug() << "Compiled kernel:";
		debug() << src.get_code();
		src.write_buffers_to_device();
		src.enqueue_range(kern, numWorkItems, workItemOffset);
		src.read_buffers_from_device();
	}

	template <class KernelType, int dimensions>
	void parallel_for(nd_range<dimensions> executionRange, KernelType kernFunctor) {
		DSELF() << "not implemented";
		detail::command::group_::check_scope();
		auto src = detail::kernel_::constructor<nd_item<dimensions>>::get(
			kernFunctor, executionRange
		);
		auto kern = src.compile();
		debug() << "Compiled kernel:";
		debug() << src.get_code();
		src.write_buffers_to_device();
		src.enqueue_nd_range(kern, executionRange);
		src.read_buffers_from_device();
	}

	// TODO: Why is the offset needed? It's already contained in the nd_range
	template <class KernelType, int dimensions>
	void parallel_for(nd_range<dimensions> numWorkItems, id<dimensions> workItemOffset, KernelType kernFunctor);


	// 3.5.3.3 Parallel For hierarchical invoke

	template <class WorkgroupFunctionType, int dimensions>
	void parallel_for_work_group(range<dimensions> numWorkGroups, WorkgroupFunctionType kernFunctor);

	template <class WorkgroupFunctionType, int dimensions>
	void parallel_for_work_group(range<dimensions> numWorkGroups, range<dimensions> workGroupSize, WorkgroupFunctionType kernFunctor);


	// OpenCL interoperability invoke

	void single_task(kernel syclKernel);

	template <int dimensions>
	void parallel_for(range<dimensions> numWorkItems, kernel syclKernel);

	template <int dimensions>
	void parallel_for(nd_range<dimensions> ndRange, kernel syclKernel);
};

} // namespace sycl
} // namespace cl
