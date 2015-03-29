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

class cmd_group {
private:
	friend class ::cl::sycl::command_group;

	// TODO: Need to deal better with threads
	SYCL_THREAD_LOCAL static command_group* last;

	template <class... Args>
	using fn = void(*)(queue*, Args...);

public:
	template<class... Args>
	static void add(fn<Args...> function, string_class name, Args... params) {
		last->commands.emplace_back(name, std::bind(function, std::placeholders::_1, params...));
	}
	static bool in_scope();
	static void check_scope(error::handler& handler = error::handler::default);

	using command_t = function_class<queue*>;
};

} // namespace detail

// A command group in SYCL as it is defined in 2.3.1 includes a kernel to be enqueued along with all the commands
// for queued data transfers that it needs in order for its execution to be successful.
class command_group {
private:
	friend class detail::cmd_group;
	using command_t = detail::cmd_group::command_t;

	vector_class<std::pair<string_class, command_t>> commands;
	queue* q;

	void enter();
	void exit();
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
