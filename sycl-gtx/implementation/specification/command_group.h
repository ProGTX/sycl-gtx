#pragma once

// 3.5.6 Command group class

#include "access.h"
#include "../common.h"
#include "../debug.h"
#include <set>

namespace cl {
namespace sycl {

// Forward declarations
class event;
class kernel;
class queue;

namespace detail {

// Forward declaration
class command_group;

namespace command {

enum class type_t {
	unspecified,
	get_accessor,
	copy_data,
	kernel
};

static debug& operator<<(debug& d, type_t t) {
	string_class str("command::type::");
	switch(t) {
		case type_t::get_accessor:
			str += "get_accessor";
			break;
		case type_t::copy_data:
			str += "copy_data";
			break;
		case type_t::kernel:
			str += "kernel";
			break;
		case type_t::unspecified:
		default:
			str += "unspecified";
			break;
	}
	d << str;
	return d;
}

struct buffer_copy {
	buffer_access buf;
	access::mode mode;
};

union metadata {
	nullptr_t empty;
	buffer_access buf_acc;
	buffer_copy buf_copy;

	metadata()
		: empty(nullptr) {}
	metadata(buffer_access buf_acc)
		: buf_acc(buf_acc) {}
	metadata(buffer_copy buf_copy)
		: buf_copy(buf_copy) {}
};
	
struct info {
	using command_f = function_class<void(queue*, vector_class<cl_event>&)>;

	string_class name;	// Only for debugging
	command_f function;
	type_t type;
	metadata data;

	static void do_nothing(queue* q, vector_class<cl_event>&) {}
};

class group_ {
private:
	friend class command_group;

	// TODO: Need to deal better with threads
	SYCL_THREAD_LOCAL static command_group* last;

	template <class... Args>
	using fn = void(*)(queue*, vector_class<cl_event>&, Args...);

public:
	// Add generic command
	template <class... Args>
	static void add(
		fn<Args...> function,
		string_class name,
		Args... params
	) {
		last->commands.push_back({
			name,
			std::bind(
				function,
				std::placeholders::_1,
				std::placeholders::_2,
				params...
			),
			type_t::unspecified
		});
	}

	// Add buffer access command
	template <bool = true>
	static void add(
		buffer_access buf_acc,
		string_class name
	) {
		last->commands.push_back({
			name,
			std::bind(
				info::do_nothing,
				std::placeholders::_1,
				std::placeholders::_2
			),
			type_t::get_accessor,
			metadata(buf_acc)
		});
		last->dependencies.insert(buf_acc.data);
	}

	// Add buffer copy command
	template <class... Args>
	static void add(
		buffer_access buf_acc,
		access::mode copy_mode,
		fn<Args...> function,
		string_class name,
		Args... params
	) {
		last->commands.push_back({
			name,
			std::bind(
				function,
				std::placeholders::_1,
				std::placeholders::_2,
				params...
			),
			type_t::copy_data,
			metadata(buffer_copy{ buf_acc, copy_mode })
		});
	}

	static bool in_scope();
	static void check_scope();

	using command_f = info::command_f;
};

} // namespace command


// A command group in SYCL as it is defined in 2.3.1 includes a kernel to be enqueued along with all the commands
// for queued data transfers that it needs in order for its execution to be successful.
class command_group {
private:
	friend class kernel;
	friend class command::group_;
	friend class queue;
	using command_f = command::group_::command_f;
	using command_t = command::info;

	vector_class<command_t> commands;
	std::set<buffer_base*> dependencies;
	queue* q;

	void enter();
	void exit();

public:
	command_group(queue* q)
		: q(q) {}

	template <typename functorT>
	command_group(functorT lambda)
		: q(nullptr) {
		enter();
		handler cgh(q);
		lambda(cgh);
		exit();
	}

	// Constructs a command group with the queue the group will enqueue its commands to
	// and a lambda function or function object containing the body of commands to enqueue.
	template <typename functorT>
	command_group(queue& primaryQueue, functorT lambda)
		: q(&primaryQueue) {
		enter();
		handler cgh(q);
		lambda(cgh);
		exit();
	}

	// TODO
	// Constructs a command group from a primary queue to be used in order to enqueue its commands to
	// and a lambda function or function object containing the body of commands to enqueue.
	// If the command group execution fails in the primary queue,
	// the SYCL runtime will try to re-schedule the whole command group to the secondary queue.
	template <typename functorT>
	command_group(queue& primaryQueue, queue& secondaryQueue, functorT lambda);

	void optimize_and_move(command_group& saveResults);
	void flush(vector_class<cl_event> wait_events);
};

} // namespace detail

} // namespace sycl
} // namespace cl
