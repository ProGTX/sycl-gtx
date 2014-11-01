#pragma once

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
		queue, buffer, image
	};
	thrower_t thrower_type;

	template<class T, thrower_t required_type>
	T* get() {
		return (thrower_type == required_type ? reinterpret_cast<T*>(thrower) : nullptr);
	}

public:
	template <class T>
	exception(T* thrower, cl_int error_code, bool is_sycl_specific = false);

	template <>
	exception(queue* thrower, cl_int error_code, bool is_sycl_specific)
		: thrower(thrower), error_code(error_code), is_sycl_specific(is_sycl_specific), thrower_type(thrower_t::queue) {}

	// TODO: Handling of buffer template parameters
	//template <>
	//exception(buffer* thrower, cl_int error_code, bool is_sycl_specific)
	//	: thrower(thrower), error_code(error_code), is_sycl_specific(is_sycl_specific), thrower_type(thrower_t::buffer) {}

	template <>
	exception(image* thrower, cl_int error_code, bool is_sycl_specific)
		: thrower(thrower), error_code(error_code), is_sycl_specific(is_sycl_specific), thrower_type(thrower_t::image) {}

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

	// Returns the buffer that caused the error.
	// Returns 0 if not a buffer error
	//buffer* get_buffer();

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

class err_handler {
public:
#ifdef __CL_ENABLE_EXCEPTIONS
	using type = error_handler;
#else
	using type = int;
#endif
	static type default_handler;

private:
	type& handler;

public:
	err_handler(type& handler);
	err_handler();
	err_handler(const err_handler&) = default;
	err_handler& operator=(const err_handler&) = default;

#if MSVC_LOW
	// Visual Studio [2013] does not support defaulted move constructors or move-assignment operators as the C++11 standard mandates.
	// http://msdn.microsoft.com/en-us/library/dn457344.aspx
	err_handler(err_handler&& move);
	err_handler& operator=(err_handler&& move);
#else
	err_handler(err_handler&&) = default;
	err_handler& operator=(err_handler&&) = default;
#endif

	static void handle(cl_int error_code, type& handler);
	void handle(cl_int error_code);
};

} // namespace helper

} // namespace sycl
} // namespace cl
