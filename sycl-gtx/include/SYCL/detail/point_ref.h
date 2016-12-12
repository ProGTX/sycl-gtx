#pragma once

#include "SYCL/detail/common.h"
#include "SYCL/detail/data_ref.h"
#include "SYCL/detail/ptr_or_val.h"

namespace cl {
namespace sycl {
namespace detail {

// Forward declaration
template <bool is_const, typename data_basic_t = ::size_t, bool holds_pointer = true>
struct point_ref;

template <bool is_const, typename data_basic_t>
struct get_value_point_t {
  using type = point_ref<is_const, data_basic_t, false>;
  static type constructor(data_basic_t&& value, data_ref::type_t type_) {
    return type(std::move(value), type_, true);
  }
  static type constructor(string_class&& value, data_ref::type_t type_) {
    return type(std::move(value), type_, true);
  }
};

template <bool is_const, typename data_basic_t, bool holds_pointer>
struct point_ref : data_ref {
protected:
  template <bool, typename, bool>
  friend struct point_ref;
  template <bool, typename>
  friend struct get_value_point_t;

  template <typename T>
  struct is_computable {
    static const bool value = !is_const && std::is_arithmetic<T>::value;
  };
  template <typename T>
  using if_is_num_assignable = typename std::enable_if<is_computable<T>::value>::type;
  template <typename T>
  using if_is_numeric = typename std::enable_if<std::is_arithmetic<T>::value>::type;

  using data_ptr_t =
    typename std::conditional<is_const, data_basic_t* const, data_basic_t*>::type;
  using data_t = typename std::remove_pointer<data_ptr_t>::type;
  using value_point_t = typename get_value_point_t<is_const, data_basic_t>::type;

  // TODO: Need to also carry references to type and name
  ptr_or_val<data_t, holds_pointer> data;

  point_ref(data_basic_t value, type_t type_, bool)
    : data_ref(get_string<data_basic_t>::get(value)), data(value) {
    type = type_;
  }
  point_ref(string_class name, type_t type_, bool)
    : data_ref(name), data(0) {
    type = type_;
  }

public:
  point_ref(data_basic_t& data, string_class name, type_t type_)
    : data_ref(name), data(&data) {
    type = type_;
  }

  operator data_basic_t() const {
    return data;
  }
  // TODO: Only allow on is_const
  operator data_basic_t&() {
    return data;
  }

  // TODO: enable_if causes here an internal MSVC error C1001
  // TODO: data_ref::operator&
  //template <class = typename std::enable_if<!is_const>::type>
  point_ref<is_const, data_basic_t*> operator&() {
    string_class name_;
    if(type == type_t::numeric) {
      name_ = name;
    }
    else {
      name_ = string_class("&(") + name + ")";
    }

    return point_ref<is_const, data_basic_t*>(&data, name_, type);
  }

  // TODO: enable_if causes here an internal MSVC error C1001
  // TODO: data_ref::operator*
  //template <
  //  class = typename std::enable_if<std::is_pointer<data_basic_t>::value>::type>
  point_ref<is_const, typename std::remove_pointer<data_basic_t>::type> operator*() {
    string_class name_;
    if(type == type_t::numeric) {
      name_ = name;
    }
    else {
      name_ = string_class("*(") + name + ")";
    }

    return point_ref<is_const, typename std::remove_pointer<data_basic_t>::type>(
      *data, name_, type
      );
  }

  template <typename T, class = if_is_num_assignable<T>>
  point_ref& operator=(T n) {
    if(type == type_t::numeric) {
      data = n;
      name = get_string<T>::get(data);
    }
    else {
      data_ref::operator=(n);
    }
    return *this;
  }

#define SYCL_POINT_REF_ARITH_ASSIGN(OP)                   \
  template <typename T, class = if_is_num_assignable<T>>  \
  point_ref& operator OP(T n) {                           \
    if(type == type_t::numeric) {                         \
      data OP n;                                          \
      name = get_string<T>::get(data);                    \
    }                                                     \
    else {                                                \
      data_ref::operator OP(n);                           \
    }                                                     \
    return *this;                                         \
  }

#define SYCL_POINT_REF_ARITH_OP(OP)                                       \
  template <typename T, class = if_is_numeric<T>>                         \
  value_point_t operator OP(T n) const {                                  \
    if(type == type_t::numeric) {                                         \
      return value_point_t(data OP n, type, true);                        \
    }                                                                     \
    else {                                                                \
      auto ret = data_ref::operator OP(n);                                \
      return value_point_t(std::move(ret.name), ret.type, true);          \
    }                                                                     \
  }                                                                       \
  template <bool is_const_, typename data_basic_t_, bool holds_pointer_>  \
  value_point_t operator OP(                                              \
    point_ref<is_const_, data_basic_t_, holds_pointer_> pref) const {     \
    if(type == type_t::numeric && pref.type == type_t::numeric) {         \
      return value_point_t(data OP pref.data, type, true);                \
    }                                                                     \
    else {                                                                \
      auto ret = data_ref::operator OP(pref);                             \
      return value_point_t(std::move(ret.name), ret.type, true);          \
    }                                                                     \
  }                                                                       \
  data_ref operator OP(data_ref dref) const {                             \
    return data_ref::operator OP(dref);                                   \
  }                                                                       \
  template <typename T, class = if_is_numeric<T>>                         \
  friend value_point_t operator OP(T n, const point_ref& rhs) {           \
    if(rhs.type == type_t::numeric) {                                     \
      return get_value_point_t<is_const, data_basic_t>::constructor(      \
        n OP rhs.data, rhs.type                                           \
      );                                                                  \
    }                                                                     \
    else {                                                                \
      auto ret = n OP (data_ref)rhs;                                      \
      return get_value_point_t<is_const, data_basic_t>::constructor(      \
        std::move(ret.name), ret.type                                     \
      );                                                                  \
    }                                                                     \
  }

  SYCL_POINT_REF_ARITH_OP(+)
  SYCL_POINT_REF_ARITH_OP(-)
  SYCL_POINT_REF_ARITH_OP(*)
  SYCL_POINT_REF_ARITH_OP(/)
  SYCL_POINT_REF_ARITH_OP(%)
  SYCL_POINT_REF_ARITH_OP(>>)
  SYCL_POINT_REF_ARITH_OP(<<)
  SYCL_POINT_REF_ARITH_OP(&)
  SYCL_POINT_REF_ARITH_OP(^)
  SYCL_POINT_REF_ARITH_OP(|)

  SYCL_POINT_REF_ARITH_ASSIGN(+=)
  SYCL_POINT_REF_ARITH_ASSIGN(-=)
  SYCL_POINT_REF_ARITH_ASSIGN(*=)
  SYCL_POINT_REF_ARITH_ASSIGN(/=)
  SYCL_POINT_REF_ARITH_ASSIGN(%=)
  SYCL_POINT_REF_ARITH_ASSIGN(>>=)
  SYCL_POINT_REF_ARITH_ASSIGN(<<=)
  SYCL_POINT_REF_ARITH_ASSIGN(&=)
  SYCL_POINT_REF_ARITH_ASSIGN(^=)
  SYCL_POINT_REF_ARITH_ASSIGN(|=)

#undef SYCL_POINT_REF_ARITH_ASSIGN
#undef SYCL_POINT_REF_ARITH_OP
};

} // namespace detail
} // namespace sycl
} // namespace cl
