#pragma once

// 2.5.7 Error handling
// 3.5 Error handling

#include "../common.h"
#include "../debug.h"
#include <memory>
#include <functional>

namespace cl {
namespace sycl {

// Forward declarations
class queue;
template <typename T, int dimensions>
struct buffer;
class image;

class exception : public std::exception {
private:
	cl_int error_code;
	bool is_sycl_specific;

	void* thrower;
	enum class thrower_t {
		other, queue, buffer, image
	};
	thrower_t thrower_type;

	template<class T, thrower_t required_type>
	T* get() {
		return (thrower_type == required_type ? reinterpret_cast<T*>(thrower) : nullptr);
	}

	exception(void* thrower, cl_int error_code, bool is_sycl_specific, thrower_t thrower_type)
		: thrower(thrower), error_code(error_code), is_sycl_specific(is_sycl_specific), thrower_type(thrower_type) {}

public:
	template <class T>
	exception(T* thrower, cl_int error_code, bool is_sycl_specific = false)
		: exception(thrower, error_code, is_sycl_specific, thrower_t::other) {}

	template <>
	exception(queue* thrower, cl_int error_code, bool is_sycl_specific)
		: exception(thrower, error_code, is_sycl_specific, thrower_t::queue) {}

	template <>
	exception(image* thrower, cl_int error_code, bool is_sycl_specific)
		: exception(thrower, error_code, is_sycl_specific, thrower_t::image) {}

	// TODO: Compiles, but linker error
	template<int dimensions>
	exception(buffer<class T, dimensions>* thrower, cl_int error_code, bool is_sycl_specific)
		: exception(thrower, error_code, is_sycl_specific, thrower_t::bufer) {}

	// Returns the OpenCL error code.
	// Returns 0 if not an OpenCL error
	cl_int get_cl_code() {
		return (is_sycl_specific ? 0 : error_code);
	}

	// Returns the SYCL-specific error code.
	// Returns 0 if not a SYCL-specific error
	cl_int get_sycl_code() {
		return (is_sycl_specific ? error_code : 0);
	}

	// Returns the queue that caused the error.
	// Returns 0 if not a queue error
	queue* get_queue() {
		return get<queue, thrower_t::queue>();
	}

	// TODO: A bit trickier than the constructor since we are forcing the type instead of deducing it
	// Returns the buffer that caused the error.
	// Returns 0 if not a buffer error
	//buffer<class T>* get_buffer() {
	//	return get<buffer<class T>, thrower_t::buffer>();
	//}

	// Returns the image that caused the error.
	// Returns 0 if not a image error
	image* get_image() {
		return get<image, thrower_t::image>();
	}
};

class error_handler {
public:
	//  called on error
	virtual void report_error(exception& error) = 0;
};

namespace helper {
namespace error {

class throw_handler : public error_handler {
public:
	virtual void report_error(exception& error) override {
		if(error.get_cl_code() != CL_SUCCESS) {
			throw error;
		}
	}
};

class code_handler : public error_handler {
private:
	cl_int& error_code;
public:
	code_handler(cl_int& error_code)
		: error_code(error_code) {}
	virtual void report_error(exception& error) override {
		error_code = error.get_cl_code();
	}
};

class async_handler : public error_handler {
private:
	// Specification isn't clear enough on this
	using function_t = std::function<void(VECTOR_CLASS<std::exception>&)>;

	friend class handler;
	function_t async_func;
	VECTOR_CLASS<std::exception> list;
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
	error_handler& actual_hndlr;
	bool is_async = false;

public:
	static throw_handler default;

	handler()
		: actual_hndlr(default) {}
	handler(cl_int& error_code)
		: hidden_hndlr(new code_handler(error_code)), actual_hndlr(*hidden_hndlr) {}
	handler(async_handler::function_t& hndlr)
		: hidden_hndlr(new async_handler(hndlr)), actual_hndlr(*hidden_hndlr), is_async(true) {}
	handler(error_handler& hndlr)
		: actual_hndlr(hndlr) {}

	handler(const handler&) = default;
	handler& operator=(const handler&) = default;

#if MSVC_LOW
private:
	void swap(handler& first, handler& second) {
		SYCL_MOVE(hidden_hndlr);
		SYCL_COPY(actual_hndlr);
		SYCL_COPY(is_async);
	}
public:
	handler(handler&& move) : actual_hndlr(move.actual_hndlr) { swap(*this, move); }
	handler operator=(handler&& move) { swap(*this, move); return *this; }
#else
	handler(handler&&) = default;
	handler operator=(handler&&) = default;
#endif

	template <class T>
	void report(T* thrower, cl_int error_code, bool is_sycl_specific = false) {
		exception e(thrower, error_code, is_sycl_specific);
		actual_hndlr.report_error(e);
	}

	void apply() {
		if(is_async) {
			((async_handler&)actual_hndlr).apply();
		}
	}
};

} // namespace error
} // namespace helper

} // namespace sycl
} // namespace cl
