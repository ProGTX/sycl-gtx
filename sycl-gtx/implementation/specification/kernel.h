#pragma once

// 3.7 Expressing parallelism through kernels
// 3.7.1 is not included here, but rather in ranges.h

#include "../common.h"
#include "../debug.h"

namespace cl {
namespace sycl {

// Forward declarations
class program;
class context;

// 3.7.2.5 Kernel class

class kernel {
private:
	friend class program;
public:
	// The default object is not valid because there is no
	// program or cl_kernel associated with it
	kernel() = delete;

	// Constructs from a valid, initialized OpenCL kernel
	kernel(cl_kernel openclKernelObject);

	// Return the OpenCL kernel object for this kernel.
	cl_kernel get() const;

	// Return the context that this kernel is defined for.
	context get_context() const;

	// Return the program that this kernel is part of.
	program get_program() const;

	// Return the name of the kernel function.
	string_class get_kernel_attributes() const;

	template<cl_int name>
	typename param_traits<cl_kernel_info, name>::param_type get_info() const;
};

} // namespace sycl
} // namespace cl
