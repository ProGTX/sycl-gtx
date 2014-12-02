#pragma once

// 3.2.6 Command group class

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
	// typename functorT: kernel functor or lambda function
	template <typename functorT>
	command_group(queue q, functorT functor) {
		enter();
		functor();
		exit();
	}
	event kernel_event() {
		DSELF() << "not implemented";
		return event();
	}
	event start_event() {
		DSELF() << "not implemented";
		return event();
	}
	event complete_event() {
		DSELF() << "not implemented";
		return event();
	}
};

} // namespace sycl
} // namespace cl
