#pragma once

// Helper functions for error handling
// 2.5.6 Error handling
// 3.6 Error handling

#include "exception.h"
#include "../common.h"
#include "../debug.h"
#include "../error_code.h"

namespace cl {
namespace sycl {

// Forward declaration
class context;

namespace detail {

struct sycl_exception : ::cl::sycl::exception {
private:
	error::code::value_t error_code;
public:
	sycl_exception(error::code::value_t error_code)
		: exception((*error::codes.find(error_code)).second), error_code(error_code) {}
};

namespace error {

class throw_handler {
public:
	virtual void report_error(exception& error) {
		debug("SYCL_ERROR::", error.what());
		throw error;
	}
	void report_error(cl_exception& error) {
		auto code = error.get_cl_code();
		if(code != CL_SUCCESS) {
			debug("SYCL_ERROR::", error.what());
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
		cl_exception e(error_code, thrower);
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
