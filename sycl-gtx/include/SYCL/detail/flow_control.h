#pragma once

#ifdef SYCL_GTX

#include "SYCL/detail/data_ref.h"
#include "SYCL/detail/common.h"
#include "SYCL/detail/debug.h"

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

// Note: Increment can only be ++ or --, other assignment doesn't work
static void for_(data_ref condition, data_ref increment) {
  kernel_::source::add<false>(string_class("for(; ") + condition.name + "; " + increment.name + ")");
}

static void break_() {
  kernel_::source::add<true>("break");
}

static void continue_() {
  kernel_::source::add<true>("continue");
}

static void return_() {
  kernel_::source::add<true>("return");
}

} // namespace control
} // namespace detail

} // namespace sycl
} // namespace cl

#define SYCL_BEGIN  \
::cl::sycl::detail::kernel_::source::add_curlies();

#define SYCL_END  \
::cl::sycl::detail::kernel_::source::remove_curlies();

#define SYCL_IF(condition)                  \
::cl::sycl::detail::control::if_(           \
  ::cl::sycl::detail::data_ref((condition)) \
);                        \
SYCL_BEGIN

#define SYCL_ELSE                     \
SYCL_END                              \
::cl::sycl::detail::control::else_(); \
SYCL_BEGIN

#define SYCL_ELSE_IF(condition)             \
SYCL_END                                    \
::cl::sycl::detail::control::else_if(       \
  ::cl::sycl::detail::data_ref((condition)) \
);                                          \
SYCL_BEGIN

  // TODO: do-while not supported
#define SYCL_WHILE(condition)               \
::cl::sycl::detail::control::while_(        \
  ::cl::sycl::detail::data_ref((condition)) \
);                                          \
SYCL_BEGIN

#define SYCL_FOR(init, condition, increment)  \
init;                                         \
::cl::sycl::detail::control::for_(            \
  ::cl::sycl::detail::data_ref((condition)),  \
  ::cl::sycl::detail::data_ref((increment))   \
);                                            \
SYCL_BEGIN

#define SYCL_BREAK  \
::cl::sycl::detail::control::break_();

#define SYCL_CONTINUE  \
::cl::sycl::detail::control::continue_();

#define SYCL_RETURN  \
::cl::sycl::detail::control::return_();

#endif // SYCL_GTX
