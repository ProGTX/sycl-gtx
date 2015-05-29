#pragma once

// 3.7 Expressing parallelism through kernels
// 3.7.1 is not included here, but rather in ranges.h

#include "context.h"
#include "error_handler.h"
#include "program.h"
#include "ranges.h"
#include "refc.h"
#include "../common.h"
#include "../debug.h"
#include <algorithm>

namespace cl {
namespace sycl {

// Forward declarations
class context;
class queue;

namespace detail {
namespace kernel_ {
	class source;
}
}

// 3.7.2.5 Kernel class

class kernel {
private:
	detail::refc<cl_kernel, clRetainKernel, clReleaseKernel> kern;
	context ctx;
	program prog;

	friend class detail::kernel_::source;

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
	context get_context() const {
		return ctx;
	}

	// Return the program that this kernel is part of.
	program get_program() const {
		return prog;
	}

private:
	template <class return_type, cl_int name>
	struct hidden {
		using real_return = return_type;
		static real_return get_info(const kernel* kern) {
			auto k = kern->get();
			real_return param_value;
			auto error_code = clGetKernelInfo(k, name, sizeof(cl_uint), &param_value, nullptr);
			detail::error::report(error_code);
			return param_value;
		}
	};
	template <cl_int name>
	struct hidden<char[], name> {
		using real_return = string_class;
		static real_return get_info(const kernel* kern) {
			auto k = kern->get();
			static const int BUFFER_SIZE = 8192;
			char param_value[BUFFER_SIZE];
			auto error_code = clGetKernelInfo(k, name, sizeof(char) * BUFFER_SIZE, param_value, nullptr);
			detail::error::report(error_code);
			return real_return(param_value);
		}
	};
	template <cl_int name>
	using param = typename param_traits<cl_kernel_info, name>::param_type;

public:
	template <cl_int name>
	typename hidden<param<name>, name>::real_return get_info() const {
		return hidden<param<name>, name>::get_info(this);
	}

	// Return the name of the kernel function
	string_class get_kernel_attributes() const {
		return get_info<CL_KERNEL_ATTRIBUTES>();
	}

	// Return the name of the kernel function
	string_class get_function_name() {
		return get_info<CL_KERNEL_FUNCTION_NAME>();
	}

private:
	void enqueue_task(queue* q) const;

	template <int dimensions>
	void enqueue_range(queue* q, range<dimensions> num_work_items, id<dimensions> offset) const {
		size_t* global_work_size = &num_work_items[0];
		size_t* offst = &((size_t&)offset[0]);

		auto error_code = clEnqueueNDRangeKernel(
			q->get(), kern.get(), dimensions,
			offst, global_work_size, nullptr,
			// TODO: Events
			0, nullptr, nullptr
		);
		detail::error::report(error_code);
	}

	template <int dimensions>
	void enqueue_nd_range(queue* q, nd_range<dimensions> execution_range) const {
		size_t* local_work_size = &execution_range.get_local_range()[0];
		size_t* offst = &((size_t&)execution_range.get_offset()[0]);

		size_t global_work_size[dimensions];
		size_t* start = &execution_range.get_global_range()[0];
		std::copy(start, start + dimensions, global_work_size);

		// Adjust global work size
		for(int i = 0; i < dimensions; ++i) {
			auto remainder = global_work_size[i] % local_work_size[i];
			if(remainder > 0) {
				global_work_size[i] += local_work_size[i] - remainder;
			}
		}

		auto error_code = clEnqueueNDRangeKernel(
			q->get(), kern.get(), dimensions,
			offst, global_work_size, local_work_size,
			// TODO: Events
			0, nullptr, nullptr
		);
		detail::error::report(error_code);
	}
};

} // namespace sycl
} // namespace cl
