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

struct command_group_ {
	// TODO: Should be thread_local
	static command_group* last;
};

} // namespace detail

// A command group in SYCL as it is defined in 2.3.1 includes a kernel to be enqueued along with all the commands
// for queued data transfers that it needs in order for its execution to be successful.
class command_group {
private:
	void enter() {
		detail::command_group_::last = this;
	}
	void exit() {
		detail::command_group_::last = nullptr;
	}
public:
	// Constructs a command group with the queue the group will enqueue its commands to
	// and a lambda function or function object containing the body of commands to enqueue.
	template <typename functorT>
	command_group(queue& primaryQueue, functorT lambda) {
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
