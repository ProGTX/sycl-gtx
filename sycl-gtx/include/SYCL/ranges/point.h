#pragma once

#include "SYCL/detail/common.h"
#include "SYCL/detail/data_ref.h"
#include "SYCL/detail/point_ref.h"

namespace cl {
namespace sycl {
namespace detail {

// Forward declarations
template <int dimensions>
struct get_special_id;
template <int dimensions>
struct get_special_range;
template <int dimensions, bool is_id>
struct identifier_code;

struct point_names {
  static const string_class id_global;
  static const string_class range_global;

  static const string_class id_local;
  static const string_class range_local;
};

#define SYCL_POINT_OP_EQ(lhs, op)             \
  for (::size_t i = 0; i < dimensions; ++i) { \
    lhs values[i] op rhs.values[i];           \
  }

template <::size_t dimensions>
struct point : data_ref {
 protected:
  friend class data_ref;
  template <::size_t>
  friend struct point;

  template <int>
  friend struct get_special_id;
  template <int>
  friend struct get_special_range;
  template <int, bool>
  friend struct identifier_code;

  static string_class name_from_type(type_t type) {
    string_class name = "";
    switch (type) {
      case type_t::id_global:
        name = point_names::id_global;
        break;
      case type_t::id_local:
        name = point_names::id_local;
        break;
      case type_t::range_global:
        name = point_names::range_global;
        break;
      case type_t::range_local:
        name = point_names::range_local;
        break;
      default:
        break;
    }
    return name;
  }

  ::size_t values[dimensions];

  void set(type_t type) {
    this->type = type;
    this->name = name_from_type(this->type);
  }

  void set(point& rhs) {
    set(rhs.type);
    SYCL_POINT_OP_EQ(this->, =);
  }

  void set(::size_t value) {
    for (::size_t i = 0; i < dimensions; ++i) {
      values[i] = value;
    }
  }

  bool is_identifier() const {
    switch (this->type) {
      case type_t::id_global:
      case type_t::id_local:
      case type_t::range_global:
      case type_t::range_local:
        return true;
      default:
        return false;
    }
  }

  point(::size_t x, ::size_t y, ::size_t z) : data_ref("") {
    this->type = type_t::numeric;
    values[0] = x;
    if (dimensions > 1) {
      values[1] = y;
      if (dimensions > 2) {
        values[2] = z;
      }
    } else {
      this->name = get_string<::size_t>::get(x);
    }
  }

 public:
// Warning: Extremely ugly,
// split in two parts to appease Clang's aversion to the equality sign
#define SYCL_POINT_ARITH_OP_P1(OP)                                      \
  data_ref operator OP(const data_ref& rhs) const {                     \
    return data_ref::operator OP(rhs);                                  \
  }                                                                     \
  point operator OP(const point& rhs) const {                           \
    point lhs(*this);                                                   \
    if (this->type == type_t::numeric && rhs.type == type_t::numeric) { \
      SYCL_POINT_OP_EQ(lhs., OP);                                       \
    }
#define SYCL_POINT_ARITH_OP_P2(OP)                                      \
  else {                                                                \
    lhs.set(type_t::general);                                           \
    static_cast<data_ref>(lhs).operator OP(rhs);                        \
  }                                                                     \
  return lhs;                                                           \
  }                                                                     \
  point& operator OP(const data_ref& rhs) {                             \
    set(type_t::general);                                               \
    return data_ref::operator OP(rhs);                                  \
  }                                                                     \
  point& operator OP(const point& rhs) {                                \
    if (this->type == type_t::numeric && rhs.type == type_t::numeric) { \
      SYCL_POINT_OP_EQ(this->, OP);                                     \
      if (dimensions == 1) {                                            \
        this->name = get_string<::size_t>::get(values[0]);              \
      }                                                                 \
    } else {                                                            \
      return operator OP((data_ref)rhs);                                \
    }                                                                   \
  }

  SYCL_POINT_ARITH_OP_P1(+)
  SYCL_POINT_ARITH_OP_P2(+=)
  SYCL_POINT_ARITH_OP_P1(-)
  SYCL_POINT_ARITH_OP_P2(-=)
  SYCL_POINT_ARITH_OP_P1(*)
  SYCL_POINT_ARITH_OP_P2(*=)
  SYCL_POINT_ARITH_OP_P1(/)
  SYCL_POINT_ARITH_OP_P2(/=)
  SYCL_POINT_ARITH_OP_P1(%)
  SYCL_POINT_ARITH_OP_P2(%=)
  SYCL_POINT_ARITH_OP_P1(>>)
  SYCL_POINT_ARITH_OP_P2(>>=)
  SYCL_POINT_ARITH_OP_P1(<<)
  SYCL_POINT_ARITH_OP_P2(<<=)
  SYCL_POINT_ARITH_OP_P1(&)
  SYCL_POINT_ARITH_OP_P2(&=)
  SYCL_POINT_ARITH_OP_P1 (^)
  SYCL_POINT_ARITH_OP_P2(^=)
  SYCL_POINT_ARITH_OP_P1(|)
  SYCL_POINT_ARITH_OP_P2(|=)

#undef SYCL_POINT_ARITH_OP_P1
#undef SYCL_POINT_ARITH_OP_P2

 private:
  template <bool is_const>
  point_ref<is_const> get_ref(int dim) {
    auto name_tmp = this->name;

    if (is_identifier()) {
      name_tmp += get_string<::size_t>::get(dim);
    } else if (this->type == type_t::numeric && name_tmp.empty()) {
      name_tmp = get_string<::size_t>::get(values[dim]);
    }

    return point_ref<is_const>(values[dim], name_tmp, this->type);
  }

 public:
  point_ref<true> get(int dim) const {
    // The const cast is ugly,
    // but the get_ref method doesn't actually modify this class
    return const_cast<point<dimensions>*>(this)  // NOLINT
        ->get_ref<true>(dim);
  }
  point_ref<false> operator[](int dim) {
    return get_ref<false>(dim);
  }
};

#undef SYCL_POINT_OP_EQ

}  // namespace detail
}  // namespace sycl
}  // namespace cl
