#pragma once

// 2.5.6 Error handling
// 3.2 Error handling

#include "../common.h"
#include "../debug.h"
#include "../error_code.h"
#include <functional>

namespace cl {
namespace sycl {

// Forward declarations
class context;

namespace detail {
namespace error {
	class handler;
}} // namespace detail::error


struct exception {
protected:
	friend class detail::error::handler;
	context* thrower;

	exception(context* thrower = nullptr)
		: thrower(thrower) {}
public:
	// Returns a descriptive string for the error, if available.
	virtual string_class get_description() const {
		return "Undefined SYCL error";
	}

	// Returns the context that caused the error.
	// Returns null if not a buffer error.
	// The pointer is to an object that is only valid as long as the exception object is valid
	context* get_context() {
		return thrower;
	}
};

struct cl_exception : exception {
private:
	friend class detail::error::handler;
	cl_int error_code;

	cl_exception(context* thrower, cl_int error_code)
		: exception(thrower), error_code(error_code) {}

	cl_exception(cl_int error_code)
		: cl_exception(nullptr, error_code) {}
public:
	cl_exception()
		: cl_exception(CL_SUCCESS) {}

	virtual string_class get_description() const override {
		return detail::error_string(error_code);
	}

	// Thrown as a result of an OpenCL API error code
	cl_int get_cl_code() const {
		return error_code;
	}
};

struct async_exception : exception {
	// stored in an exception_list for asynchronous errors
};

// TODO: The exception_ptr class is used to store cl::sycl::exception objects
// and allows exception objects to be transferred between threads.
// It is equivalent to the std::exception_ptr class.
typedef exception* exception_ptr;

class exception_list : public exception {
	// Used as a container for a list of asynchronous exceptions
public:
	typedef exception_ptr value_type;
	typedef const value_type& reference;
	typedef const value_type& const_reference;
	typedef size_t size_type;
	//typedef /*unspecified* iterator;
	//typedef /*unspecified* const_iterator;
	size_t size() const;
	//iterator begin() const; // first asynchronous exception
	//iterator end() const; // lasst asynchronous exception
};


namespace detail {

struct sycl_exception : ::cl::sycl::exception {
private:
	error::code::value_t error_code;
public:
	sycl_exception(error::code::value_t error_code)
		: exception(nullptr) {}
	virtual string_class get_description() const override {
		return (*error::codes.find(error_code)).second;
	}
};

namespace error {

class throw_handler {
public:
	virtual void report_error(exception& error) {
		debug("SYCL_ERROR::", error.get_description());
		throw error;
	}
	void report_error(cl_exception& error) {
		auto code = error.get_cl_code();
		if(code != CL_SUCCESS) {
			debug("SYCL_ERROR::", error.get_description());
			throw error;
		}
	}
};

class handler {
private:
	shared_ptr_class<throw_handler> hidden_hndlr;
	bool is_async = false;
	context* thrower = nullptr;

public:
	static handler default;

	handler()
		: hidden_hndlr(new throw_handler()) {}
	handler(context& thrower)
		: thrower(&thrower), hidden_hndlr(new throw_handler()) {}

	// Copy and move semantics
	handler(const handler&) = default;
#if MSVC_LOW
	handler(handler&& move)
		: SYCL_MOVE_INIT(hidden_hndlr), is_async(move.is_async), thrower(move.thrower)
	{}
	friend void swap(handler& first, handler& second) {
		using std::swap;
		SYCL_SWAP(hidden_hndlr);
		SYCL_SWAP(is_async);
		SYCL_SWAP(thrower);
	}
#else
	handler(handler&&) = default;
#endif

	void report(cl_int error_code) const {
		cl_exception e(thrower, error_code);
		hidden_hndlr->report_error(e);
	}
	void report(code::value_t error_code) const {
		sycl_exception e(error_code);
		hidden_hndlr->report_error(e);
	}

	void set_thrower(context* thrower) {
		this->thrower = thrower;
	}

	void apply() {
		if(is_async) {
			// TODO
			//((async_handler*)actual_hndlr)->apply();
		}
	}
};

struct report {
	template<class T>
	report(T* caller, cl_int error_code) {
		caller->handler.report(error_code);
	}
	template<class T>
	report(T* caller, code::value_t error_code) {
		caller->handler.report(error_code);
	}
	report(code::value_t error_code) {
		handler h;
		h.report(error_code);
	}
};

} // namespace error
} // namespace detail

} // namespace sycl
} // namespace cl
