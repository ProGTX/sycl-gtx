#pragma once

// 3.3.3 Context class

#include "device.h"
#include "refc.h"
#include "device_selector.h"
#include "error_handler.h"
#include "info.h"
#include "param_traits2.h"
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
	async_handler asyncHandler;
	friend struct detail::error::thrower;

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
		: SYCL_MOVE_INIT(ctx), SYCL_MOVE_INIT(target_devices), SYCL_MOVE_INIT(asyncHandler) {}
	friend void swap(context& first, context& second) {
		using std::swap;
		SYCL_SWAP(ctx);
		SYCL_SWAP(target_devices);
		SYCL_SWAP(asyncHandler);
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
	template <class return_t, info::context param>
	struct traits {
		static return_t get(const context* contex) {
			return_t param_value;
			detail::get_cl_info<info::context, param, sizeof(return_t)>(
				contex->ctx.get(), &param_value
			);
			return param_value;
		}
	};
	template <typename Contained, info::context param>
	struct traits<vector_class<Contained>, param> : detail::traits<Contained> {
		static Container get(const context* contex) {
			Contained param_value[BUFFER_SIZE];
			size_t actual_size;
			detail::get_cl_info<info::context, param, BUFFER_SIZE * type_size>(
				contex->ctx.get(), param_value, &actual_size
			);
			return Container(param_value, param_value + actual_size / type_size);
		}
	};

public:
	// Queries OpenCL information for the underlying cl_context
	template <info::context param>
	typename param_traits2<info::context, param>::type
	get_info() const {
		return traits<typename param_traits2<info::context, param>::type, param>::get(this);
	}
};

} // namespace sycl
} // namespace cl
