#pragma once

// 3.2.6 Command group class

#include "queue.h"

namespace cl {
namespace sycl {

// A command group in SYCL as it is defined in 2.3.1 includes a kernel to be enqueued along with all the commands
// for queued data transfers that it needs in order for its execution to be successful.

// typename functorT: kernel functor or lambda function
template <typename functorT>
class command_group {
public:
	command_group(queue q, functorT functor) {}
};

} // namespace sycl
} // namespace cl
