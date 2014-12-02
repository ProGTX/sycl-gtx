#pragma once

// 2.5.6 Error handling
// 3.2 Error handling

#include "../common.h"
#include "../debug.h"
#include "../error_code.h"
#include <memory>
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
	string_class get_description() {
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


class error_handler {
public:
	//  called on error
	virtual void report_error(exception& error) = 0;
};

namespace detail {
namespace error {

class throw_handler : public error_handler {
public:
	virtual void report_error(exception& error) override {
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

class code_handler : public error_handler {
private:
	cl_int& error_code;
	friend class handler;
public:
	code_handler(cl_int& error_code)
		: error_code(error_code) {}
	virtual void report_error(exception& error) override {
		error_code = ((cl_exception&)error).get_cl_code();
	}
};

class async_handler : public error_handler {
public:
	// Specification isn't clear enough on this
	using function_t = std::function<void(vector_class<std::exception>&)>;
private:
	friend class handler;
	function_t async_func;
	vector_class<std::exception> list;
public:
	async_handler(function_t async_func)
		: async_func(async_func) {}
	virtual void report_error(exception& error) override {
		return report_error((std::exception&)error);
	}
	void report_error(std::exception& error) {
		list.push_back(error);
	}
	void apply() {
		async_func(list);
		list.clear();
	}
};

class handler {
private:
	std::shared_ptr<error_handler> hidden_hndlr;
	error_handler* actual_hndlr;
	bool is_async = false;
	context* thrower = nullptr;

public:
	static throw_handler default;

	// TODO: Add thrower to constructors
	handler()
		: actual_hndlr(&default) {}
	handler(cl_int& error_code)
		: hidden_hndlr(new code_handler(error_code)), actual_hndlr(hidden_hndlr.get()) {}
	handler(async_handler::function_t& hndlr)
		: hidden_hndlr(new async_handler(hndlr)), actual_hndlr(hidden_hndlr.get()), is_async(true) {}
	handler(error_handler& hndlr)
		: actual_hndlr(&hndlr) {}

	// Copy and move semantics
	handler(const handler&) = default;
#if MSVC_LOW
	handler(handler&& move)
		: SYCL_MOVE_INIT(hidden_hndlr), SYCL_MOVE_INIT(actual_hndlr), is_async(move.is_async)
	{}
	friend void swap(handler& first, handler& second) {
		using std::swap;
		SYCL_SWAP(hidden_hndlr);
		SYCL_SWAP(actual_hndlr);
		SYCL_SWAP(is_async);
	}
#else
	handler(handler&&) = default;
#endif

private:
	void report(cl_int error_code, bool is_sycl_specific) {
		cl_exception e(thrower, error_code);
		actual_hndlr->report_error(e);
	}
public:
	void report(cl_int error_code) {
		report(error_code, false);
	}
	void report(detail::error::code::value_t error_code) {
		report(error_code, true);
	}
	void report() {
		cl_exception e;
		bool right_type = true;
		try {
			e = cl_exception(thrower, ((code_handler*)actual_hndlr)->error_code);
		}
		catch(std::bad_cast&) {
			right_type = false;
			// TODO: Do something else?
		}
		if(right_type) {
			actual_hndlr->report_error(e);
		}
	}

	void set_thrower(context* thrower) {
		this->thrower = thrower;
	}

	void apply() {
		if(is_async) {
			((async_handler*)actual_hndlr)->apply();
		}
	}

	error_handler& get() {
		return *actual_hndlr;
	}
};

} // namespace error
} // namespace detail

} // namespace sycl
} // namespace cl
