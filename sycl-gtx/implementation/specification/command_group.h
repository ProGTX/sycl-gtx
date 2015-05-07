#pragma once

// 3.5.6 Command group class

#include "buffer.h"
#include "event.h"
#include "queue.h"
#include "../common.h"
#include "../debug.h"

namespace cl {
namespace sycl {

namespace detail {
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
	using command_f = function_class<queue*>;

	string_class name;	// Only for debugging
	command_f function;
	type_t type;
	metadata data;

	static void do_nothing(queue* q) {}
};

class group_ {
private:
	friend class ::cl::sycl::command_group;

	// TODO: Need to deal better with threads
	SYCL_THREAD_LOCAL static command_group* last;

	template <class... Args>
	using fn = void(*)(queue*, Args...);

public:
	// Add generic command
	template<class... Args>
	static void add(
		fn<Args...> function,
		string_class name,
		Args... params
	) {
		last->commands.push_back({
			name,
			std::bind(function, std::placeholders::_1, params...),
			type_t::unspecified
		});
	}

	// Add buffer access command
	template<bool = true>
	static void add(
		buffer_access buf_acc,
		string_class name
	) {
		last->commands.push_back({
			name,
			std::bind(info::do_nothing, std::placeholders::_1),
			type_t::get_accessor,
			metadata(buf_acc)
		});
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
			std::bind(function, std::placeholders::_1, params...),
			type_t::copy_data,
			metadata(buffer_copy{ buf_acc, copy_mode })
		});
	}

	static bool in_scope();
	static void check_scope(error::handler& handler = error::handler::default);

	using command_f = info::command_f;
};

} // namespace command
} // namespace detail

// A command group in SYCL as it is defined in 2.3.1 includes a kernel to be enqueued along with all the commands
// for queued data transfers that it needs in order for its execution to be successful.
class command_group {
private:
	friend class detail::command::group_;
	using command_f = detail::command::group_::command_f;

	vector_class<detail::command::info> commands;
	queue* q;

	void enter();
	void exit();
	void optimize();
	void flush();
public:
	// Constructs a command group with the queue the group will enqueue its commands to
	// and a lambda function or function object containing the body of commands to enqueue.
	template <typename functorT>
	command_group(queue& primaryQueue, functorT lambda)
		: q(&primaryQueue) {
		enter();
		lambda();
		exit();
	}

	// TODO
	// Constructs a command group from a primary queue to be used in order to enqueue its commands to
	// and a lambda function or function object containing the body of commands to enqueue.
	// If the command group execution fails in the primary queue,
	// the SYCL runtime will try to re-schedule the whole command group to the secondary queue.
	template <typename functorT>
	command_group(queue& primaryQueue, queue& secondaryQueue, functorT lambda);

	~command_group();

	// Return the event object that the command group waits on to begin execution.
	event start_event() {
		DSELF() << "not implemented";
		return event();
	}

	// Return the event representing completion of the command group's kernel.
	event kernel_event() {
		DSELF() << "not implemented";
		return event();
	}

	// Return the event representing completion of the entire command group
	// including any required data movement commands.
	event complete_event() {
		DSELF() << "not implemented";
		return event();
	}
};

} // namespace sycl
} // namespace cl
