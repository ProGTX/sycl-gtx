#pragma once

#include "../common.h"

namespace cl {
namespace sycl {

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
