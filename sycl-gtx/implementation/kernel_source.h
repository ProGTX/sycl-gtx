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

namespace kernel_ {

// Forward declaration
template<class Input>
struct constructor;


class source {
private:
	struct buf_info {
		buffer_access acc;
		string_class resource_name;
		string_class type_name;
		size_t size;
	};

	static const string_class resource_name_root;
	SYCL_THREAD_LOCAL static int num_resources;

	string_class tab_offset;

	static int num_kernels;
	int kernel_id;

	string_class kernel_name;
	vector_class<string_class> lines;
	string_class final_code;
	std::unordered_map<void*, buf_info> resources;

	// TODO: Multithreading support
	SYCL_THREAD_LOCAL static source* scope;

	template<class Input>
	friend struct constructor;

	string_class generate_accessor_list() const;

	static void compile_command(queue* q, source src, shared_unique<kernel> kern);
	static void enqueue_task_command(queue* q, shared_unique<kernel> kern);

	static void enter(source& src);
	static source exit(source& src);

	source()
		:	tab_offset("\t"),
			kernel_id(++num_kernels),
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
	static string_class register_resource(const accessor_core<DataType, dimensions, mode, target>& acc) {
		if(scope == nullptr) {
			//error::report(error::code::NOT_IN_KERNEL_SCOPE);
			return "";
		}

		string_class resource_name;
		auto buf = (buffer<DataType, dimensions>*) acc.resource();
		auto it = scope->resources.find(buf);

		if(it == scope->resources.end()) {
			resource_name = resource_name_root + std::to_string(++num_resources);
			scope->resources[buf] = {
				{ buf, mode, target },
				resource_name,
				detail::type_string<DataType>() + '*',
				acc.argument_size()
			};
		}
		else {
			resource_name = it->second.resource_name;
		}

		return resource_name;
	}

	template <bool auto_end = true>
	static void add(string_class line) {
		scope->lines.push_back(scope->tab_offset + line + (auto_end ? ';' : ' '));
	}

	static void add_curlies() {
		add<false>("{");
		scope->tab_offset.push_back('\t');
	}
	static void remove_curlies() {
		scope->tab_offset.pop_back();
		add<false>("}");
	}

	static string_class get_name(access::target target);
};

} // namespace kernel_
} // namespace detail

} // namespace sycl
} // namespace cl
