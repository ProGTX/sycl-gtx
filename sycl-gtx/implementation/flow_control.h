#pragma once

#include "data_ref.h"
#include "common.h"
#include "debug.h"

namespace cl {
namespace sycl {

namespace detail {
namespace control {

static void open_curly() {
	kernel_::source::add<false>("{");
}

static void close_curly() {
	kernel_::source::add<false>("}");
}
	
template <class T>
static void if_(T condition) {
	kernel_::source::add<false>(
		string_class("if(") + data_ref::get_name(condition) + ")"
	);
}

template <class T>
static void else_if(T condition) {
	kernel_::source::add<false>(
		string_class("else if(") + data_ref::get_name(condition) + ")"
	);
}

static void else_() {
	kernel_::source::add<false>("else");
}

} // namespace control
} // namespace detail

} // namespace sycl
} // namespace cl

#define SYCL_IF(condition) \
	::cl::sycl::detail::control::if_((condition));

#define SYCL_BLOCK(code)						\
	{											\
	::cl::sycl::detail::control::open_curly();	\
	code										\
	::cl::sycl::detail::control::close_curly();	\
	}

#define SYCL_THEN(code) \
	SYCL_BLOCK(code)

#define SYCL_ELSE(code)						\
	::cl::sycl::detail::control::else_();	\
	SYCL_BLOCK(code)

#define SYCL_ELSE_IF(condition) \
	::cl::sycl::detail::control::else_if((condition));
