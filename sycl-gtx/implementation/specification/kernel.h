#pragma once

// 3.7 Expressing parallelism through kernels
// 3.7.1 is not included here, but rather in ranges.h

#include "context.h"
#include "error_handler.h"
#include "program.h"
#include "refc.h"
#include "../common.h"
#include "../debug.h"

namespace cl {
namespace sycl {

// Forward declarations
class context;

// 3.7.2.5 Kernel class

class kernel {
private:
	refc::ptr<cl_kernel> kern;
	context ctx;
	program prog;

public:
	// The default object is not valid because there is no
	// program or cl_kernel associated with it
	kernel() = delete;
	kernel(nullptr_t) = delete;

	// Constructs from a valid, initialized OpenCL kernel
	kernel(cl_kernel openclKernelObject);

	// Return the OpenCL kernel object for this kernel.
	cl_kernel get() const {
		return kern.get();
	}

	// Return the context that this kernel is defined for.
	context get_context() const;

	// Return the program that this kernel is part of.
	program get_program() const;

	// Return the name of the kernel function.
	string_class get_kernel_attributes() const;


private:
	template<class return_type, cl_int name>
	struct hidden {
		using real_return = return_type;
		static real_return get_info(kernel* kern) {
			auto k = kern->get();
			real_return param_value;
			auto error_code = clGetKernelInfo(k, name, sizeof(cl_uint), &param_value, nullptr);
			//kern->handler.report(contex, error_code);
			return param_value;
		}
	};
	template<cl_int name>
	struct hidden<char[], name> {
		using real_return = string_class;
		static real_return get_info(kernel* kern) {
			auto k = kern->get();
			static const int BUFFER_SIZE = 8192;
			char param_value[BUFFER_SIZE];
			auto error_code = clGetKernelInfo(k, name, sizeof(char) * BUFFER_SIZE, param_value, nullptr);
			//kern->handler.report(error_code);
			return real_return(param_value);
		}
	};
	template<cl_int name>
	using param = typename param_traits<cl_kernel_info, name>::param_type;

public:
	template<cl_int name>
	typename hidden<param<name>, name>::real_return get_info() {
		return hidden<param<name>, name>::get_info(this);
	}
};

} // namespace sycl
} // namespace cl
