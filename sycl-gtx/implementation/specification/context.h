#pragma once

// 3.3.3 Context class

#include "device.h"
#include "refc.h"
#include "device_selector.h"
#include "error_handler.h"
#include "info.h"
#include "param_traits.h"
#include "../common.h"
#include "../debug.h"

namespace cl {
namespace sycl {

// Forward declarations
class platform;
class program;

// 2.3.1, point 2
// Any OpenCL resource that is acquired by the user is attached to a context.
// A context contains a collection of devices that the host can use
// and manages memory objects that can be shared between the devices.
// Data movement between devices within a context may be efficient and hidden by the underlying runtime
// while data movement between contexts must involve the host.
// A given context can only wrap devices owned by a single platform.
class context {
private:
	detail::refc<cl_context, clRetainContext, clReleaseContext> ctx;
	vector_class<device> target_devices;

	// Master constructor
	context(
		cl_context c,
		const async_handler& asyncHandler,
		info::gl_context_interop interopFlag,
		vector_class<device> deviceList = {},
		const platform* plt = nullptr,
		const device_selector& deviceSelector = *(device_selector::default)
	);
public:
	// Default constructor that chooses the context according the heuristics of the default selector.
	// Returns synchronous errors via the SYCL exception class.
	context();

	// Constructs a context object for SYCL host using an async_handler for handling asynchronous errors.
	explicit context(const async_handler& asyncHandler);

	// Executes a retain on the cl_context
	context(cl_context clContext, const async_handler& asyncHandler = detail::default_async_handler);

	context(const device_selector& deviceSelector, info::gl_context_interop interopFlag = false, const async_handler& asyncHandler = detail::default_async_handler);

	context(const device& dev, info::gl_context_interop interopFlag = false, const async_handler& asyncHandler = detail::default_async_handler);

	context(const platform& plt, info::gl_context_interop interopFlag = false, const async_handler& asyncHandler = detail::default_async_handler);

	context(vector_class<device> deviceList, info::gl_context_interop interopFlag = false, const async_handler& asyncHandler = detail::default_async_handler);

	// Copy and move semantics
	context(const context&) = default;
#if MSVC_LOW
	context(context&& move)
		: SYCL_MOVE_INIT(ctx), SYCL_MOVE_INIT(target_devices) {}
	friend void swap(context& first, context& second) {
		using std::swap;
		SYCL_SWAP(ctx);
		SYCL_SWAP(target_devices);
	}
#else
	context(context&&) = default;
#endif

public:
	// Returns the underlying cl context object, after retaining the cl_context.
	cl_context get() const;

	// TODO: Specifies whether the context is in SYCL Host Execution Mode
	bool is_host() const;

	// Returns the SYCL platform that the context is initialized for.
	platform get_platform();

	// Returns the set of devices that are part of this context.
	vector_class<device> get_devices() const;

private:
	template<class return_type, cl_int name>
	struct hidden {
		using real_return = return_type;
		static real_return get_info(const context* contex) {
			auto c = contex->ctx.get();
			real_return param_value;
			auto error_code = clGetContextInfo(c, name, sizeof(real_return),& param_value, nullptr);
			//contex->handler.report(contex, error_code);
			return param_value;
		}
	};
	template<class return_type, cl_int name>
	struct hidden<return_type[], name> {
		using real_return = vector_class<return_type>;
		static real_return get_info(const context* contex) {
			auto c = contex->ctx.get();
			static const int BUFFER_SIZE = 1024;
			return_type param_value[BUFFER_SIZE];
			std::size_t actual_size;
			std::size_t type_size = sizeof(return_type);
			auto error_code = clGetContextInfo(c, name, BUFFER_SIZE * type_size, param_value,& actual_size);
			//contex->handler.report(error_code);
			return real_return(param_value, param_value + actual_size / type_size);
		}
	};
	template<cl_int name>
	using param = typename param_traits<cl_context_info, name>::param_type;
public:

	// Queries OpenCL information for the underlying cl_context.
	template<cl_int name>
	typename hidden<param<name>, name>::real_return get_info() const {
		return hidden<param<name>, name>::get_info(this);
	}
};

} // namespace sycl
} // namespace cl
