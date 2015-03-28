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

// Forward declaration
template<class Input>
struct constructor;


class source {
private:
	struct tuple {
		string_class type_name;
		access::mode mode;
		access::target target;
		buffer_base* buffer;
	};

	static int num_kernels;
	int kernel_id;

	string_class kernel_name;
	vector_class<string_class> lines;
	string_class final_code;
	std::unordered_map<string_class, tuple> resources;

	// TODO: Multithreading support
	SYCL_THREAD_LOCAL static source* scope;

	template<class Input>
	friend struct constructor;

	string_class generate_accessor_list() const;
	static string_class get_name(access::target target);
	template<typename DataType>
	static string_class get_name() {
		// TODO
		return "int*";
	}

	static void compile_command(queue* q, source src, shared_unique<kernel> kern);
	static void enqueue_task_command(queue* q, shared_unique<kernel> kern);

	source()
		:	kernel_id(++num_kernels),
			kernel_name(string_class("_sycl_kernel_") + std::to_string(kernel_id)) {}

public:
	string_class get_code();
	shared_unique<kernel> compile() const;
	void write_buffers_to_device() const;
	void enqueue_task(shared_unique<kernel> kern) const;
	void read_buffers_from_device() const;

	template<int dimensions>
	void enqueue_range(shared_unique<kernel> kern, range<dimensions> num_work_items) const {
		DSELF() << "not implemented";
	}


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

// Single task invoke
template<>
struct constructor<void> {
	static source get(function_class<void> kern) {
		source src;
		source::scope = &src;

		kern(); // MSVC2013 complains about this, but compiles and links.

		source::scope = nullptr;
		return src;
	}
};

// Parallel For
template<int dimensions>
struct constructor<id<dimensions>> {
	static source get(function_class<id<dimensions>> kern) {
		source src;
		source::scope = &src;

		// TODO: id
		id<dimensions> id_(0);
		kern(id_);

		source::scope = nullptr;
		return src;
	}
};

} // namespace kernel_
} // namespace detail

} // namespace sycl
} // namespace cl
