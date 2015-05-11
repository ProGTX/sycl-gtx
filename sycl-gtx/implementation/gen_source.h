#pragma once

#include "specification\access.h"
#include "specification\accessor\buffer.h"
#include "specification\command_group.h"
#include "common.h"
#include "debug.h"
#include <unordered_map>


namespace cl {
namespace sycl {

// Forward declarations
class kernel;
class queue;

namespace detail {

namespace type_name {

template <typename DataType>
static string_class get();

#define STRING_TYPE_FUNCION(type)	\
template<>							\
static string_class get<type>() {	\
	return #type "*";				\
}

STRING_TYPE_FUNCION(int)
STRING_TYPE_FUNCION(float)

#undef STRING_TYPE_FUNCION

} // namespace type_name


namespace kernel_ {

// Forward declaration
template<class Input>
struct constructor;


class source {
private:
	struct buf_info {
		buffer_access acc;
		string_class type_name;
	};

	static int num_kernels;
	int kernel_id;

	string_class kernel_name;
	vector_class<string_class> lines;
	string_class final_code;
	std::unordered_map<string_class, buf_info> resources;

	// TODO: Multithreading support
	SYCL_THREAD_LOCAL static source* scope;

	template<class Input>
	friend struct constructor;

	string_class generate_accessor_list() const;

	static void compile_command(queue* q, source src, shared_unique<kernel> kern);
	static void enqueue_task_command(queue* q, shared_unique<kernel> kern);

	source()
		:	kernel_id(++num_kernels),
			kernel_name(string_class("_sycl_kernel_") + std::to_string(kernel_id)) {}

public:
	static bool in_scope();

	string_class get_code();
	shared_unique<kernel> compile() const;
	void write_buffers_to_device() const;
	void read_buffers_from_device() const;

	void enqueue_task(shared_unique<kernel> kern) const;

	template <int dimensions>
	static void enqueue_range_command(
		queue* q, detail::shared_unique<kernel> kern, range<dimensions> num_work_items, id<dimensions> offset
	) {
		(*kern)->enqueue_range(q, num_work_items, offset);
	}
	template <int dimensions>
	void enqueue_range(shared_unique<kernel> kern, range<dimensions> num_work_items, id<dimensions> offset) const {
		command::group_::add(enqueue_range_command, __func__, kern, num_work_items, offset);
	}

	template <int dimensions>
	static void enqueue_nd_range_command(
		queue* q, detail::shared_unique<kernel> kern, nd_range<dimensions> execution_range
		) {
		(*kern)->enqueue_nd_range(q, execution_range);
	}
	template <int dimensions>
	void enqueue_nd_range(shared_unique<kernel> kern, nd_range<dimensions> execution_range) const {
		command::group_::add(enqueue_nd_range_command, __func__, kern, execution_range);
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
			scope->resources[name] = { { buf, mode, target }, type_name::get<DataType>() };
		}
	}

	template <bool auto_end = true>
	static void add(string_class line) {
		scope->lines.push_back('\t' + line + (auto_end ? ';' : ' '));
	}

	static string_class get_name(access::target target);
};

} // namespace kernel_
} // namespace detail

} // namespace sycl
} // namespace cl
