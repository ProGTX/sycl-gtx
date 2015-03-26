#pragma once

#include "specification\access.h"
#include "specification\accessor.h"
#include "specification\command_group.h"
#include "specification\ranges.h"
#include "common.h"
#include "debug.h"
#include <unordered_map>


namespace cl {
namespace sycl {

// Forward declarations
class kernel;
class queue;

namespace detail {

namespace kernel_ {

class source {
private:
	struct tuple {
		string_class type_name;
		access::mode mode;
		access::target target;
		buffer_base* buffer;
	};

	string_class kernelName;
	vector_class<string_class> lines;
	std::unordered_map<string_class, tuple> resources;

	// TODO: Multithreading support
	SYCL_THREAD_LOCAL static source* scope;

	string_class generate_accessor_list() const;
	static string_class get_name(access::target target);
	template<typename DataType>
	static string_class get_name() {
		// TODO
		return "int*";
	}

	static void compile_command(queue* q, source src, shared_unique<kernel> kern);
	static void enqueue_task_command(queue* q, shared_unique<kernel> kern);

public:
	template<class KernelType>
	source(string_class kernelName, KernelType kern)
		: kernelName(kernelName) {
		scope = this;
		kern();
		scope = nullptr;
	}

	string_class get_code();
	shared_unique<kernel> compile() const;
	void write_buffers_to_device() const;
	void enqueue_task(shared_unique<kernel> kern) const;
	void read_buffers_from_device() const;

	template <typename DataType, int dimensions, access::mode mode, access::target target>
	static void register_resource(const accessor_core<DataType, dimensions, mode, target>& acc) {
		if(scope == nullptr) {
			//error::report(error::code::NOT_IN_KERNEL_SCOPE);
			return;
		}

		auto name = acc.get_resource_name();
		auto it = scope->resources.find(name);

		if(it == scope->resources.end()) {
			auto buf = (buffer<DataType, dimensions>*) acc.resource();
			scope->resources[name] = { get_name<DataType>(), mode, target, buf };
		}
	}

	// TODO: Should be better hidden
	static void add(string_class line) {
		scope->lines.push_back('\t' + line + ';');
	}

	template <int dimensions>
	static string_class to_string(id<dimensions> index) {
		// TODO
		return "0";
	}
	template <>
	static string_class to_string(id<1> index) {
		return std::to_string(index[0]);
	}
};

} // namespace kernel_
} // namespace detail

} // namespace sycl
} // namespace cl
