#pragma once

// 3.7.2.6 Program class

#include "error_handler.h"
#include "param_traits.h"
#include "refc.h"
#include "../common.h"

namespace cl {
namespace sycl {

// Forward declarations
class kernel;
class context;
class device;
class queue;

namespace detail {
namespace kernel_ {
	class source;
}
}


class program {
protected:
	friend class detail::kernel_::source;

	detail::refc<cl_program, clRetainProgram, clReleaseProgram> prog;

	program(string_class source, queue* q);
public:
	// Creates an empty program object for all devices associated with context
	program(const context& context) {}
	
	// Creates an empty program object devices in list associated with the context
	program(const context& context, vector_class<device> device_list);

	// Creates a program object from a cl_program object
	program(const context& context, cl_program clProgram) {}

	// Creates a program by linking a list of other programs
	program(vector_class<program> program_list, string_class link_options = "");

	// Obtains a SYCL program object from a SYCL kernel name and compiles it ready-to-link
	template<typename kernelT>
	void compile_from_kernel_name(string_class compile_options = "");
	
	// Obtains a SYCL program object from a SYCL kernel name and builds it ready-to-run
	template<typename kernelT>
	void build_from_kernel_name(string_class compile_options = "");
	
	// Gets a kernel from a given name (Functor)
	template<typename kernelT>
	kernel get_kernel() const;
	
	template<cl_int name> typename
	param_traits<cl_program_info, name>::param_type get_info() const;

	vector_class<vector_class<char>> get_binaries() const;
	vector_class<::size_t> get_binary_sizes() const;
	vector_class<device> get_devices() const;
	string_class get_build_options() const;

	cl_program get() const {
		return prog.get();
	}
};

} // namespace sycl
} // namespace cl
