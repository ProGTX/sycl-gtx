#pragma once

// 3.5.5 Program class

#include "error_handler.h"
#include "info.h"
#include "param_traits2.h"
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
	bool linked = false;

	program(string_class source, queue* q);
public:
	// Creates an empty program object for all devices associated with context
	explicit program(const context& context) {}
	
	// Creates an empty program object for all devices in list associated with the context
	program(const context& context, vector_class<device> deviceList);

	// Creates a program object from a cl_program object
	program(const context& context, cl_program clProgram) {}

	// Creates a program by linking a list of other programs
	program(vector_class<program> programList, string_class linkOptions = "");

	// TODO: Somehow provide the ability to compile and build programs

	// Obtains a SYCL program object from a SYCL kernel name and compiles it ready-to-link
	template <typename kernelT>
	void compile_from_kernel_name(string_class compile_options = "");
	// Obtains a SYCL program object from a SYCL kernel name and builds it ready-to-run
	template <typename kernelT>
	void build_from_kernel_name(string_class compile_options = "");
	// Gets a kernel from a given name (Functor)
	template <typename kernelT>
	kernel get_kernel() const;

	void link(string_class linking_options = "");

	bool is_linked() const {
		return linked;
	}

private:
	template <typename ReturnType, info::program param>
	struct traits
		: detail::array_traits<
			ReturnType,
			info::program,
			param,
			detail::traits_buffer_default<ReturnType>::size
		> {
		return_t get_info(const program* p) {
			return get(p->prog.get());
		}
	};

	template <typename Contained_, info::program param>
	struct traits<vector_class<Contained_>, param>
		: detail::array_traits<Contained_, info::program, param> {
		Container get_info(const program* p) {
			get(p->prog.get());
			return Container(param_value, param_value + actual_size / type_size);
		}
	};

	template <class Contained_>
	struct traits<vector_class<vector_class<Contained_>>, info::program::binaries>
		: detail::array_traits<Contained_*, info::program, info::program::binaries> {
		using DoubleContainer = vector_class<vector_class<Contained_>>;
		DoubleContainer get_info(const program* p) {
			auto binary_sizes = p->get_info<info::program::binary_sizes>();
			get(p->prog.get());

			DoubleContainer ret;
			static const auto inner_type_size = sizeof(Contained_);
			size_t i = 0;
			for(auto bin_size : binary_sizes) {
				ret.emplace_back(param_value[i], param_value[i] + bin_size / inner_type_size);
				++i;
			}
			return ret;
		}
	};

public:
	template <info::program param>
	typename param_traits2<info::program, param>::type get_info() const {
		return traits<param_traits2_t<info::program, param>, param>().get_info(this);
	}

	vector_class<vector_class<unsigned char>> get_binaries() const;
	vector_class<size_t> get_binary_sizes() const;
	vector_class<device> get_devices() const;
	string_class get_build_options() const;

	cl_program get() const {
		return prog.get();
	}
};

} // namespace sycl
} // namespace cl
