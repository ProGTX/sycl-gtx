#pragma once

// 3.5.4 Context class

#include "device.h"
#include "refc.h"
#include "device_selector.h"
#include "error_handler.h"
#include "param_traits.h"
#include "../common.h"
#include "../debug.h"

namespace cl {
namespace sycl {

// Forward declarations
class platform;
class program;

// Used as the notification function for contexts.
class context_notify {
private:
	friend class context;
	static void CL_CALLBACK forward(const char* errinfo, const void* private_info, size_t cb, void* caller) {
		static_cast<context_notify*>(caller)->operator()(errinfo, private_info, cb);
	}
public:
	virtual void operator()(const string_class errinfo, const void* private_info, size_t cb) = 0;
	virtual void operator()(program t_program) = 0;
};

// 2.3.1, point 2
// Any OpenCL resource that is acquired by the user is attached to a context.
// A context contains a collection of devices that the host can use
// and manages memory objects that can be shared between the devices.
// Data movement between devices within a context may be efficient and hidden by the underlying runtime
// while data movement between contexts must involve the host.
// A given context can only wrap devices owned by a single platform.
class context {
private:
	refc::ptr<cl_context> ctx;
	vector_class<device> target_devices;

	detail::error::handler handler;
	static detail::error::handler& default_error;

	static refc::ptr<cl_context> reserve(cl_context c = nullptr);

	// Master constructor
	context(
		cl_context c,
		const cl_context_properties* properties,
		vector_class<device> target_devices = {},
		const device_selector& dev_sel = *(device_selector::default),
		detail::error::handler& handler = default_error,
		platform* plt = nullptr,
		context_notify* ctx_notify = nullptr
	);
public:
	// Chooses the context according to the heuristics of the default selector
	context();

	// Executes a retain on the cl_context
	explicit context(cl_context context);

	// Constructs with a single device retrieved from the provided device selector object.
	context(const device_selector& deviceSelector, cl_context_properties* properties = nullptr);

	// Constructs context from device
	context(const device& dev, cl_context_properties* properties = nullptr);

	// Constructs context from platform
	context(const platform& plt, cl_context_properties* properties = nullptr);

	// Constructs context from a list of devices
	context(vector_class<device> deviceList, cl_context_properties* properties = nullptr);


	// Same constructors as above, just with an asynchronous error handler

	template<class... Args>
	context(function_class<Args...>& async_handler)
		: context(nullptr, nullptr, {}, *(device_selector::default), detail::error::async_handler<Args...>(async_handler)) {}

	template<class... Args>
	context(const device_selector& deviceSelector, cl_context_properties* properties, function_class<Args...>& async_handler);

	template<class... Args>
	context(const device& dev, cl_context_properties* properties, function_class<Args...>& async_handler);

	template<class... Args>
	context(const platform& plt, cl_context_properties* properties, function_class<Args...>& async_handler);

	template<class... Args>
	context(vector_class<device> deviceList, cl_context_properties* properties, function_class<Args...>& async_handler);

	// Copy and move semantics
	context(const context&) = default;
#if MSVC_LOW
	context(context&& move)
		: SYCL_MOVE_INIT(ctx), SYCL_MOVE_INIT(target_devices), SYCL_MOVE_INIT(handler) {}
	friend void swap(context& first, context& second) {
		using std::swap;
		SYCL_SWAP(ctx);
		SYCL_SWAP(target_devices);
		SYCL_SWAP(handler);
	}
#else
	context(context&&) = default;
#endif

public:
	// Returns the underlying cl context object, after retaining the cl_context.
	cl_context get() const;

	// TODO: Specifies whether the context is in SYCL Host Execution Mode
	bool is_host() const;

	vector_class<device> get_devices() const;

private:
	template<class return_type, cl_int name>
	struct hidden {
		using real_return = return_type;
		static real_return get_info(const context* contex) {
			auto c = contex->ctx.get();
			real_return param_value;
			auto error_code = clGetContextInfo(c, name, sizeof(real_return), &param_value, nullptr);
			contex->handler.report(contex, error_code);
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
			auto error_code = clGetContextInfo(c, name, BUFFER_SIZE * type_size, param_value, &actual_size);
			contex->handler.report(error_code);
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
