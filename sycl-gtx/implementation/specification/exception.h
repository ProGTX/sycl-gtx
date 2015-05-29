#pragma once

// 3.6.2 Exception Class Interface

#include "../common.h"
#include "../error_code.h"

namespace cl {
namespace sycl {

// Forward declarations
class context;
namespace detail {
namespace error {
	class handler;
}
}

struct exception {
protected:
	friend class detail::error::handler;
	string_class description;
	context* thrower;

	exception(string_class description, context* thrower = nullptr)
		: description(description), thrower(thrower) {}
public:
	exception()
		: exception("Undefined SYCL Error") {}

	// Returns a descriptive string for the error, if available.
	string_class what() {
		return description;
	}

	// Returns the context that caused the error.
	// Returns null if not a buffer error.
	context* get_context() {
		return thrower;
	}
};

struct cl_exception : exception {
private:
	friend class detail::error::handler;
	cl_int error_code;

	cl_exception(cl_int error_code, context* thrower = nullptr)
		: exception(detail::error_string(error_code), thrower), error_code(error_code) {}
public:
	cl_exception()
		: cl_exception(CL_SUCCESS, nullptr) {}

	// Thrown as a result of an OpenCL API error code
	cl_int get_cl_code() const {
		return error_code;
	}
};

struct async_exception : exception {
	// stored in an exception_list for asynchronous errors
};


// TODO
using unspecified_t = int;
using exception_ptr = unspecified_t;

// TODO: Used as a container for a list of asynchronous exceptions
class exception_list {
public:
	using value_type = exception_ptr;
	using reference = value_type&;
	using const_reference = const value_type&;
	using size_type = size_t;
	using iterator = unspecified_t;
	using const_iterator = unspecified_t;

	size_t size() const;
	iterator begin() const; // first asynchronous exception
	iterator end() const; // refer to past-the-end last asynchronous exception
};

using async_handler = function_class<void(cl::sycl::exception_list)>;

// TODO
class runtime_error : public exception {};
class kernel_error : public runtime_error {};
class accessor_error : public runtime_error {};
class nd_range_error : public runtime_error {};
class event_error : public runtime_error {};
class invalid_parameter_error : public runtime_error {};
class device_error : public exception {};
class compile_program_error : public device_error {};
class link_program_error : public device_error {};
class invalid_object_error : public device_error {};
class memory_allocation_error : public device_error {};
class platform_error : public device_error {};
class profiling_error : public device_error {};
class feature_not_supported : public device_error {};

} // namespace sycl
} // namespace cl
