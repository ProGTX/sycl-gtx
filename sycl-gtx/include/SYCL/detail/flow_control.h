#pragma once

#ifdef SYCL_GTX

#include "SYCL/detail/common.h"
#include "SYCL/detail/data_ref.h"
#include "SYCL/detail/debug.h"

namespace cl {
namespace sycl {

namespace detail {
namespace control {

static void if_detail(data_ref condition) {
  kernel_ns::source::add<false>(string_class("if(") + condition.name + ")");
}

static void else_if(data_ref condition) {
  kernel_ns::source::add<false>(string_class("else if(") + condition.name +
                                ")");
}

static void else_detail() {
  kernel_ns::source::add<false>("else");
}

static void while_detail(data_ref condition) {
  kernel_ns::source::add<false>(string_class("while( ") + condition.name + ")");
}

/** Note: Increment can only be ++ or --, other assignments don't work */
static void for_detail(data_ref condition, data_ref increment) {
  kernel_ns::source::add<false>(string_class("for(; ") + condition.name + "; " +
                                increment.name + ")");
}

static void break_detail() {
  kernel_ns::source::add<true>("break");
}

static void continue_detail() {
  kernel_ns::source::add<true>("continue");
}

static void return_detail() {
  kernel_ns::source::add<true>("return");
}

}  // namespace control
}  // namespace detail

}  // namespace sycl
}  // namespace cl

#define SYCL_BEGIN ::cl::sycl::detail::kernel_ns::source::add_curlies();

#define SYCL_END ::cl::sycl::detail::kernel_ns::source::remove_curlies();

#define SYCL_IF(condition)                        \
  ::cl::sycl::detail::control::if_detail(         \
      ::cl::sycl::detail::data_ref((condition))); \
  SYCL_BEGIN

#define SYCL_ELSE                             \
  SYCL_END                                    \
  ::cl::sycl::detail::control::else_detail(); \
  SYCL_BEGIN

#define SYCL_ELSE_IF(condition)                   \
  SYCL_END                                        \
  ::cl::sycl::detail::control::else_if(           \
      ::cl::sycl::detail::data_ref((condition))); \
  SYCL_BEGIN

// TODO(progtx): do-while not supported
#define SYCL_WHILE(condition)                     \
  ::cl::sycl::detail::control::while_detail(      \
      ::cl::sycl::detail::data_ref((condition))); \
  SYCL_BEGIN

#define SYCL_FOR(init, condition, increment)      \
  init;                                           \
  ::cl::sycl::detail::control::for_detail(        \
      ::cl::sycl::detail::data_ref((condition)),  \
      ::cl::sycl::detail::data_ref((increment))); \
  SYCL_BEGIN

#define SYCL_BREAK ::cl::sycl::detail::control::break_detail();

#define SYCL_CONTINUE ::cl::sycl::detail::control::continue_detail();

#define SYCL_RETURN ::cl::sycl::detail::control::return_detail();

#endif  // SYCL_GTX
