#pragma once

#include "kernel.h"
#include "ranges.h"
#include "../common.h"
#include "../gen_source.h"
#include "../invoke_source.h"

// 3.5.3 SYCL functions for invoking kernels

namespace cl {
namespace sycl {

namespace detail {

// Used to determine first argument in a functor
// https://functionalcpp.wordpress.com/2013/08/05/function-traits/
template <class T>
struct first_arg;

// First function argument
template <class R, class... Args>
struct first_arg<R(Args...)> {
	using type = typename std::tuple_element<0, std::tuple<Args...>>::type;
};

// Member function pointer
template <class C, class R, class... Args>
struct first_arg<R(C::*)(Args...)> : public first_arg<R(Args...)>
{};

// const member function pointer
template <class C, class R, class... Args>
struct first_arg<R(C::*)(Args...) const> : public first_arg<R(Args...)>
{};

// Functor
template <class F>
struct first_arg {
	using type = typename first_arg<decltype(&F::operator())>::type;
};

} // namespace detail


// 3.5.3.4 Command group handler class
class handler {
private:
	friend class command_group;
	// TODO: Implementation defined constructor
	handler() {}

public:
	void set_arg(int arg_index, detail::accessor_base acc_obj);
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
	void parallel_for(range<dimensions> num_work_items, KernelType kernFunctor) {
		parallel_for(num_work_items, id<dimensions>(), kernFunctor);
	}

	// This type of kernel can be invoked with a function accepting either an id or an item as parameter
	template <class KernelType, int dimensions>
	void parallel_for(range<dimensions> num_work_items, id<dimensions> work_item_offset, KernelType kernFunctor) {
		detail::command::group_::check_scope();
		auto src = detail::kernel_::constructor<typename detail::first_arg<KernelType>::type>::get(
			kernFunctor, num_work_items, work_item_offset
		);
		auto kern = src.compile();
		debug() << "Compiled kernel:";
		debug() << src.get_code();
		src.write_buffers_to_device();
		src.enqueue_range(kern, num_work_items, work_item_offset);
		src.read_buffers_from_device();
	}

	template <class KernelType, int dimensions>
	void parallel_for(nd_range<dimensions> execution_range, KernelType kernFunctor) {
		DSELF() << "not implemented";
		detail::command::group_::check_scope();
		auto src = detail::kernel_::constructor<nd_item<dimensions>>::get(
			kernFunctor, execution_range
		);
		auto kern = src.compile();
		debug() << "Compiled kernel:";
		debug() << src.get_code();
		src.write_buffers_to_device();
		src.enqueue_nd_range(kern, execution_range);
		src.read_buffers_from_device();
	}
};

/*

template <typename KernelName, class KernelType>
void single_task(KernelType);

template <typename KernelName, class KernelType, int dimensions>
void parallel_for(range<dimensions> num_work_items, KernelType);

template <typename KernelName, class KernelType, int dimensions>
void parallel_for(range<dimensions> num_work_items, id<dimensions> work_item_offset, KernelType);

template <typename KernelName, class KernelType, int dimensions>
void parallel_for(nd_range<dimensions> execution_range, KernelType);

template <class KernelName, class WorkgroupFunctionType, int dimensions>
void parallel_for_work_group(range<dimensions> num_work_groups, WorkgroupFunctionType);

template <class KernelType, int dimensions>
void parallel_for_work_item(group num_work_items, KernelType);

*/

} // namespace sycl
} // namespace cl
