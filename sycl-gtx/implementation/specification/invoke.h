#pragma once

// 3.7.3 Invoking kernels

#include "access.h"
#include "../common.h"
#include "../debug.h"

namespace cl {
namespace sycl {

// Forward declaration
template <typename DataType, int dimensions, access::mode mode, access::target target>
class accessor;

namespace detail {

// Forward declaration
class accessor_base;

namespace kernel_ {

class source {
private:
	string_class KernelName;
	vector_class<string_class> src;
	vector_class<std::tuple<access::mode, access::target, accessor_base*>> resources;

	// TODO: Multithreading support
	static source* scope;

	template<class KernelType>
	source(string_class KernelName, KernelType kern)
		: KernelName(KernelName) {}

	void execute();
	string_class get_source() {
		DSELF() << "not implemented";
		return "";
	}

public:
	template <typename DataType, int dimensions, access::mode mode, access::target target>
	static void add(accessor<DataType, dimensions, mode, target>& acc) {
		if(scope == nullptr) {
			error::report(acc, error::code::NOT_IN_KERNEL_SCOPE);
			return;
		}

		scope->resources.emplace_back(mode, target, &acc);
	}
	
	template<class KernelType>
	static string_class generate(string_class KernelName, KernelType kern) {
		auto src = source(KernelName, kern);
		source::scope = &src;
		src.execute();
		source::scope = nullptr;
		return src.get_source();
	}
};

} // namespace kernel_
} // namespace detail


// TODO: Passing kernel names
// Will need to divert slightly from the specification
// Diversion could be avoided if I could get functor name at compile time

template<class KernelType>
void single_task(string_class KernelName, KernelType kern) {
	detail::cmd_group::check_scope();
	auto src = detail::kernel_::source::generate(KernelName, kern);
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
