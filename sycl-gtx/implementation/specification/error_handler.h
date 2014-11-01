#pragma once

// 2.5.7 Error handling
// 3.5 Error handling

#include "../common.h"
#include "../debug.h"

namespace cl {
namespace sycl {

// Forward declarations
class queue;
template <typename T, int dimensions>
struct buffer;
class image;

class exception {
private:
	cl_int error_code;
	bool is_sycl_specific;

	void* thrower;
	enum class thrower_t {
		otherm, queue, buffer, image
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
		throw error;
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

} // namespace error
} // namespace helper

} // namespace sycl
} // namespace cl
