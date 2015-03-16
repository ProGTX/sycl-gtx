#pragma once

#include "specification\access.h"
#include "specification\accessor.h"
#include "common.h"
#include "debug.h"
#include <vector>


namespace cl {
namespace sycl {

namespace detail {

// Forward declaration
class accessor_base;


namespace kernel_ {

class source {
private:
	struct tuple {
		string_class name;
		string_class type;
		access::mode mode;
		access::target target;
	};

	string_class kernelName;
	vector_class<string_class> lines;
	std::vector<tuple> resources;

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
	static void add(const accessor_core<DataType, dimensions, mode, target>& acc) {
		if(scope == nullptr) {
			//error::report(error::code::NOT_IN_KERNEL_SCOPE);
			return;
		}

		scope->resources.push_back({ acc.resource_name(), get_name<DataType>(), mode, target });
	}

	// TODO: Should be better hidden
	static void add(string_class line) {
		scope->lines.push_back('\t' + line + ';');
	}

	template<class KernelType>
	static string_class generate(string_class kernelName, KernelType kern) {
		source src(kernelName, kern);
		return src.get();
	}
};

} // namespace kernel_




} // namespace detail

} // namespace sycl
} // namespace cl
