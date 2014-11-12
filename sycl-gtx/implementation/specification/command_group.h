#pragma once

// 3.2.6 Command group class

#include "event.h"
#include "queue.h"
#include "../common.h"
#include "../debug.h"

namespace cl {
namespace sycl {

namespace detail {

struct command_group_interface {
	virtual event kernel_event() {
		debug::warning(__func__) << "Should not be called.";
		return event();
	}
	virtual event start_event() {
		debug::warning(__func__) << "Should not be called.";
		return event();
	}
	virtual event complete_event() {
		debug::warning(__func__) << "Should not be called.";
		return event();
	}
};

// This class is the actual implementation of the command_group
// Three classes are needed as a workaround to the inability to deduce templated class constructor arguments
template <typename functorT>
class command_group : public command_group_interface {
public:
	command_group(queue q, functorT functor) {
		DSELF() << "not implemented";
	}
	virtual event kernel_event() override {
		DSELF() << "not implemented";
		return event();
	}
	virtual event start_event() override {
		DSELF() << "not implemented";
		return event();
	}
	virtual event complete_event() override {
		DSELF() << "not implemented";
		return event();
	}
};

} // namespace detail


// A command group in SYCL as it is defined in 2.3.1 includes a kernel to be enqueued along with all the commands
// for queued data transfers that it needs in order for its execution to be successful.
class command_group {
private:
	detail::command_group_interface group;
public:
	// typename functorT: kernel functor or lambda function
	template <typename functorT>
	command_group(queue q, functorT functor)
		: group(detail::command_group<functorT>(q, functor)) {}
	event kernel_event() {
		group.kernel_event();
	}
	event start_event() {
		group.start_event();
	}
	event complete_event() {
		group.complete_event();
	}
};

} // namespace sycl
} // namespace cl
