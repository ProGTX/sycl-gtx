#pragma once

// 2.5.7 Error handling
// 3.5 Error handling

#include "../common.h"
#include "../debug.h"
#include "../error_code.h"
#include <memory>
#include <functional>

namespace cl {
namespace sycl {

// Forward declarations
class queue;
template <typename DataType, int dimensions>
struct buffer;
class image;

namespace detail {
namespace error {
class handler;
}
}

class exception : public std::exception {
private:
	friend class detail::error::handler;

	cl_int error_code = 0;
	bool is_sycl_specific_ = false;

	void* thrower = nullptr;
	enum class thrower_t {
		other, queue, buffer, image
	};
	thrower_t thrower_type = thrower_t::other;

	template<class T, thrower_t required_type>
	T* get() const {
		return (thrower_type == required_type ? reinterpret_cast<T*>(thrower) : nullptr);
	}

	exception(void* thrower, cl_int error_code, bool is_sycl_specific, thrower_t thrower_type)
		: thrower(thrower), error_code(error_code), is_sycl_specific_(is_sycl_specific), thrower_type(thrower_type) {}

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

	exception() {}
public:

	bool is_sycl_specific() const {
		return is_sycl_specific_;
	}

	// Returns the OpenCL error code.
	// Returns 0 if not an OpenCL error
	cl_int get_cl_code() const {
		return (is_sycl_specific_ ? 0 : error_code);
	}

	// Returns the SYCL-specific error code.
	// Returns 0 if not a SYCL-specific error
	cl_int get_sycl_code() const {
		return (is_sycl_specific_ ? error_code : 0);
	}

	// Returns the queue that caused the error.
	// Returns 0 if not a queue error
	queue* get_queue() const {
		return get<queue, thrower_t::queue>();
	}

	// TODO: A bit trickier than the constructor since we are forcing the type instead of deducing it
	// Returns the buffer that caused the error.
	// Returns 0 if not a buffer error
	//buffer<class T>* get_buffer() const {
	//	return get<buffer<class T>, thrower_t::buffer>();
	//}

	// Returns the image that caused the error.
	// Returns 0 if not an image error
	image* get_image() const {
		return get<image, thrower_t::image>();
	}

#if MSVC_LOW
	virtual const char* what() const override {
#else
	virtual const char* what() const noexcept override {
#endif
		return (
			is_sycl_specific_ ?
			detail::error::codes.find((detail::error::code::value_t)error_code)->second.data() :
			detail::error_string(error_code)
		);
	}
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
		auto code = (error.is_sycl_specific() ? error.get_sycl_code() : error.get_cl_code());
		if(code != CL_SUCCESS) {
			debug("SYCL_ERROR::", error.what());
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
		error_code = error.get_cl_code();
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

	void* thrower = this;
	using thrower_t = exception::thrower_t;
	thrower_t thrower_type = thrower_t::other;

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
		exception e(thrower, error_code, is_sycl_specific, thrower_type);
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
		exception e;
		bool right_type = true;
		try {
			e = exception(thrower, ((code_handler*)actual_hndlr)->error_code, false, thrower_type);
		}
		catch(std::bad_cast&) {
			right_type = false;
			// TODO: Do something else?
		}
		if(right_type) {
			actual_hndlr->report_error(e);
		}
	}

	template <class T>
	void set_thrower(T* thrower);

	template <>
	void set_thrower(queue* thrower) {
		this->thrower = thrower;
		thrower_type = thrower_t::queue;
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
