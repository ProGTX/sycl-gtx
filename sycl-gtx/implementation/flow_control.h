#pragma once

#include "data_ref.h"
#include "common.h"
#include "debug.h"

namespace cl {
namespace sycl {

namespace detail {
namespace control {
	
static void if_(data_ref condition) {
	kernel_::source::add<false>(
		string_class("if(") + condition.name + ")"
	);
}

static void else_if(data_ref condition) {
	kernel_::source::add<false>(
		string_class("else if(") + condition.name + ")"
	);
}

static void else_() {
	kernel_::source::add<false>("else");
}

static void while_(data_ref condition) {
	kernel_::source::add<false>(string_class("while( ") + condition.name + ")");
}

} // namespace control
} // namespace detail

} // namespace sycl
} // namespace cl

#define SYCL_IF(condition)						\
::cl::sycl::detail::control::if_(				\
	::cl::sycl::detail::data_ref((condition))	\
);

#define SYCL_BEGIN	\
::cl::sycl::detail::kernel_::source::add_curlies();

#define SYCL_END	\
::cl::sycl::detail::kernel_::source::remove_curlies();

#define SYCL_BLOCK(code)	\
{							\
SYCL_BEGIN					\
code						\
SYCL_END					\
}

#define SYCL_THEN(code) \
SYCL_BLOCK(code)

#define SYCL_ELSE(code)					\
::cl::sycl::detail::control::else_();	\
SYCL_BLOCK(code)

#define SYCL_ELSE_IF(condition)					\
::cl::sycl::detail::control::else_if(			\
	::cl::sycl::detail::data_ref((condition))	\
);

#define SYCL_WHILE(condition)					\
::cl::sycl::detail::control::while_(			\
	::cl::sycl::detail::data_ref((condition))	\
);
