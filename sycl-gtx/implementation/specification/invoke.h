#pragma once

// 3.7.3 Invoking kernels

#include "access.h"
#include "../common.h"
#include "../debug.h"
#include <unordered_map>

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
	// C++14 would come in handy here with addressing tuples by type
	struct tuple {
		string_class type;
		access::mode mode;
		access::target target;
	};

	string_class kernelName;
	vector_class<string_class> lines;
	std::unordered_map<const accessor_base*, tuple> resources;

	// TODO: Multithreading support
	SYCL_THREAD_LOCAL static source* scope;

	template<class KernelType>
	source(string_class kernelName, KernelType kern)
		: kernelName(kernelName) {
		scope = this;
		kern();
		scope = nullptr;
	}

	string_class get();
	string_class generate_accessor_list();
	static string_class get_name(access::target target);
	template<typename DataType>
	static string_class get_name() {
		// TODO
		return "int*";
	}

public:
	template <typename DataType, int dimensions, access::mode mode, access::target target>
	static void add(const accessor<DataType, dimensions, mode, target>* const acc) {
		if(scope == nullptr) {
			//error::report(error::code::NOT_IN_KERNEL_SCOPE);
			return;
		}

		auto accessor_it = scope->resources.find(acc);
		if(accessor_it != scope->resources.end()) {
			accessor_it->second = { get_name<DataType>(), mode, target };
		}
	}
	
	template<class KernelType>
	static string_class generate(string_class kernelName, KernelType kern) {
		auto src = source(kernelName, kern);
		return src.get();
	}
};

} // namespace kernel_
} // namespace detail


// TODO: Passing kernel names
// Will need to divert slightly from the specification
// Diversion could be avoided if I could get functor name at compile time

template<class KernelType>
void single_task(string_class kernelName, KernelType kern) {
	detail::cmd_group::check_scope();
	auto src = detail::kernel_::source::generate(kernelName, kern);
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
