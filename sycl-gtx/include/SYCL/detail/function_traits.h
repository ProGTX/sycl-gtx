#pragma once

// Determining function traits at compile time

#include <tuple>

namespace cl {
namespace sycl {
namespace detail {

/**
 * Used to determine first argument in a functor
 *
 * https://functionalcpp.wordpress.com/2013/08/05/function-traits/
 */
template <class T>
struct first_arg;

/** First function argument */
template <class R, class... Args>
struct first_arg<R(Args...)> {
  using type = typename std::tuple_element<0, std::tuple<Args...>>::type;
};

/** No function arguments */
template <class R>
struct first_arg<R(void)> {
  using type = void;
};

/** Member function pointer */
template <class C, class R, class... Args>
struct first_arg<R (C::*)(Args...)> : public first_arg<R(Args...)> {};

/** const member function pointer */
template <class C, class R, class... Args>
struct first_arg<R (C::*)(Args...) const> : public first_arg<R(Args...)> {};

/** Functor */
template <class F>
struct first_arg {
  using type = typename first_arg<decltype(&F::operator())>::type;
};

}  // namespace detail
}  // namespace sycl
}  // namespace cl
