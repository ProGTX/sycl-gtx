#pragma once

// 3.6 Program class

#include "../common.h"
#include "context.h"
#include "device.h"

namespace cl {
namespace sycl {

// A program contains one or more kernels and any functions or libraries necessary for the program’s execution.
// A program will be enqueued inside a context and each of the kernels will be enqueued on a corresponding device.
class program {
	template <typename T>
	program(context dev_context, device target_dev);

	program(context dev_context, device target_dev, STRING_CLASS kernel_string);
};

} // namespace sycl
} // namespace cl
