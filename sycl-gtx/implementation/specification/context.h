#pragma once

// 3.5.4 Context class

#include "refc.h"
#include "device_selector.h"
#include "error_handler.h"
#include "param_traits.h"
#include "../common.h"
#include "../debug.h"

namespace cl {
namespace sycl {

// Forward declarations
class device;
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
	unsigned int best_device_id = 0;

	detail::error::handler handler;
	static error_handler& default_error;

	static refc::ptr<cl_context> reserve(cl_context c = nullptr);
	static vector_class<device> load_devices();

	// Master constructor
	context(
		cl_context c,
		const cl_context_properties* properties,
		vector_class<device> target_devices = {},
		const device_selector& dev_sel = *(device_selector::default),
		error_handler& handler = default_error,
		platform* plt = nullptr,
		context_notify* ctx_notify = nullptr
	);
public:
	context();
	explicit context(cl_context context);
	context(const device_selector& deviceSelector, cl_context_properties* properties = nullptr);
	context(const device& dev, cl_context_properties* properties = nullptr);
	context(const platform& plt, cl_context_properties* properties = nullptr);
	context(vector_class<device> deviceList, cl_context_properties* properties = nullptr);

	/* constructors with asynchronous error handler supplied */

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
		: SYCL_MOVE_INIT(ctx), SYCL_MOVE_INIT(target_devices), SYCL_MOVE_INIT(handler), best_device_id(move.best_device_id) {}
	friend void swap(context& first, context& second) {
		using std::swap;
		SYCL_SWAP(ctx);
		SYCL_SWAP(target_devices);
		SYCL_SWAP(handler);
		SYCL_SWAP(best_device_id);
	}
#else
	context(context&&) = default;
#endif

public:
	cl_context get() const;

private:
	template<class return_type, cl_int name>
	struct hidden {
		using real_return = return_type;
		static real_return get_info(context* contex) {
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
		static real_return get_info(context* contex) {
			auto c = contex->ctx.get();
			static const int BUFFER_SIZE = 1024;
			return_type param_value[BUFFER_SIZE];
			std::size_t actual_size;
			std::size_t type_size = sizeof(return_type);
			auto error_code = clGetContextInfo(c, name, BUFFER_SIZE * type_size, &param_value, &actual_size);
			contex->handler.report(error_code);
			return real_return(param_value, param_value + actual_size / type_size);
		}
	};
	template<cl_int name>
	using param = typename param_traits<cl_context_info, name>::param_type;
public:
	template<cl_int name>
	typename hidden<param<name>, name>::real_return get_info() {
		return hidden<param<name>, name>::get_info(this);
	}
};

} // namespace sycl
} // namespace cl
