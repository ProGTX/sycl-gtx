#pragma once

// 3.7.2 Vector types
// B.5 vec class

#include "base.h"
#include "helpers.h"
#include "vec_members.h"
#include "cl_vec.h"
#include "../access.h"
#include "../../common.h"
#include "../../counter.h"
#include "../../data_ref.h"


namespace cl {
namespace sycl {

template <typename dataT, int numElements>
class vec :
  public detail::vectors::base<dataT, numElements>,
  public detail::vectors::members<dataT, numElements> {
private:
  template <typename, int, access::mode, access::target, typename>
  friend class detail::accessor_;
  template <int, typename, int, access::mode, access::target>
  friend class detail::accessor_device_ref;
  template <typename, int>
  friend class detail::vectors::base;

  using Base = detail::vectors::base<dataT, numElements>;
  using Members = detail::vectors::members<dataT, numElements>;

  using genvector = detail::vectors::cl_base<dataT, numElements, numElements>;
  using data_ref = detail::data_ref;
  using type_t = data_ref::type_t;

  // Helper constructor to help with assignment
  vec(const string_class& name, bool, bool)
    : Base(name, true), Members(this) {}

  template <typename T>
  vec& assign(const T& copy) {
    if(this->type == type_t::expression) {
      vec b(this->name, true, true);
      this->name = std::move(b.name);
      this->type = type_t::general;
    }
    Base::operator=(copy);
    return *this;
  }

  vec(string_class name_, type_t type = type_t::general)
    : Base(name_), Members(this) {
    this->type = type;
  }

public:
  vec()
    : Base(), Members(this) {}
  vec(const vec& copy)
    : Base(copy.name, true), Members(this) {}
  vec(const data_ref& copy)
    : Base(copy.name, true), Members(this) {}
  vec(vec&& move)
    : Base(std::move(move.name)), Members(this) {
    this->type = move.type;
  }
  vec(data_ref&& move)
    : Base(std::move(move.name), true), Members(this) {}

  vec& operator=(const vec& copy) {
    return assign((Base)copy);
  }
  vec& operator=(const data_ref& copy) {
    return assign(copy);
  }
  vec& operator=(vec&& move) {
    return assign((Base)move);
  }
  vec& operator=(data_ref&& move) {
    return assign(move);
  }
  vec& operator=(const dataT& n) {
    return assign(n);
  }

  template <int num = numElements>
  vec(const data_ref& x, const data_ref& y, SYCL_ENABLE_IF_DIM(2))
    : Base(x, y), Members(this) {}
  template <int num = numElements>
  vec(const data_ref& x, const data_ref& y, const data_ref& z, SYCL_ENABLE_IF_DIM(3))
    : Base(x, y, z), Members(this) {}

  // TODO
  operator genvector() const {
    return genvector();
  }

  // TODO: Operators
#define SYCL_VEC_OP(op)                                 \
  vec operator op(const vec& v) const {                 \
    auto r = data_ref::operator op(v);                  \
    return vec(std::move(r.name), type_t::expression);  \
  }                                                     \
  vec operator op(const data_ref& d) const {            \
    auto r = data_ref::operator op(d);                  \
    return vec(std::move(r.name), type_t::expression);  \
  }

  SYCL_VEC_OP(+)
    SYCL_VEC_OP(-)
    SYCL_VEC_OP(*)

#undef SYCL_VEC_OP

};


template <typename dataT>
class vec<dataT, 1> :
  public detail::vectors::base<dataT, 1>,
  public detail::vectors::members<dataT, 1> {
private:
  template <typename, int, access::mode, access::target, typename>
  friend class detail::accessor_;
  template <int, typename, int, access::mode, access::target>
  friend class detail::accessor_device_ref;
  template <typename, int>
  friend class detail::vectors::base;

  using Base = detail::vectors::base<dataT, 1>;
  using Members = detail::vectors::members<dataT, 1>;

  using genvector = detail::vectors::cl_base<dataT, 1, 1>;
  using data_ref = detail::data_ref;
  using type_t = data_ref::type_t;

  // Helper constructor to help with assignment
  vec(const string_class& name, bool, bool)
    : Base(name, true), Members(this) {}

  template <typename T>
  vec& assign(const T& copy) {
    if(this->type == type_t::expression) {
      vec b(this->name, true, true);
      this->name = std::move(b.name);
      this->type = type_t::general;
    }
    Base::operator=(copy);
    return *this;
  }

  vec(string_class name_, type_t type = type_t::general)
    : Base(name_), Members(this) {
    this->type = type;
  }

public:
  vec()
    : Base(), Members(this) {}
  vec(const vec& copy)
    : Base(copy.name, true), Members(this) {}
  vec(const data_ref& copy)
    : Base(copy.name, true), Members(this) {}
  vec(vec&& move)
    : Base(std::move(move.name)), Members(this) {
    this->type = move.type;
  }
  vec(data_ref&& move)
    : Base(std::move(move.name), true), Members(this) {}
  vec(const dataT& n)
    : Base(detail::get_string<dataT>::get(n), true), Members(this) {}

  vec& operator=(const vec& copy) {
    return assign((Base)copy);
  }
  vec& operator=(const data_ref& copy) {
    return assign(copy);
  }
  vec& operator=(vec&& move) {
    return assign((Base)move);
  }
  vec& operator=(data_ref&& move) {
    return assign(move);
  }
  vec& operator=(const dataT& n) {
    return assign(n);
  }

  // TODO
  operator genvector() const {
    return genvector();
  }

  // TODO: Operators
#define SYCL_VEC_OP(op)                                 \
  vec operator op(const data_ref& d) const {            \
    auto r = data_ref::operator op(d);                  \
    return vec(std::move(r.name), type_t::expression);  \
  }

  SYCL_VEC_OP(+)
    SYCL_VEC_OP(-)
    SYCL_VEC_OP(*)

#undef SYCL_VEC_OP

};


// 3.10.1 Description of the built-in types available for SYCL host and device

#define  SYCL_VEC_SCALAR(base)                            \
  using base##1 = vec<base, 1>;                           \
  using cl_##base = detail::vectors::cl_base<base, 1, 1>;

#define  SYCL_VEC_USCALAR(base)                                     \
  SYCL_VEC_SCALAR(base)                                             \
  using u##base##1 = vec<unsigned base, 1>;                         \
  using cl_u##base = detail::vectors::cl_base<unsigned base, 1, 1>;

#define  SYCL_VEC_VECTOR(base, num)            \
  using base##num = vec<base, num>;          \
  using cl_##base##num = detail::vectors::cl_base<base, num, num>;

#define  SYCL_VEC_UVECTOR(base, num)                                        \
  SYCL_VEC_VECTOR(base, num)                                                \
  using u##base##num = vec<unsigned base, num>;                             \
  using cl_u##base##num = detail::vectors::cl_base<unsigned base, num, num>;

#define SYCL_ADD_VEC_VECTOR(base) \
  SYCL_VEC_SCALAR(base)           \
  SYCL_VEC_VECTOR(base, 2)        \
  SYCL_VEC_VECTOR(base, 3)        \
  SYCL_VEC_VECTOR(base, 4)        \
  SYCL_VEC_VECTOR(base, 8)        \
  SYCL_VEC_VECTOR(base, 16)

#define SYCL_ADD_VEC_UVECTOR(base)  \
  SYCL_VEC_USCALAR(base)            \
  SYCL_VEC_UVECTOR(base, 2)         \
  SYCL_VEC_UVECTOR(base, 3)         \
  SYCL_VEC_UVECTOR(base, 4)         \
  SYCL_VEC_UVECTOR(base, 8)         \
  SYCL_VEC_UVECTOR(base, 16)

SYCL_VEC_SCALAR(bool)
SYCL_ADD_VEC_UVECTOR(int)
SYCL_ADD_VEC_UVECTOR(char)
SYCL_ADD_VEC_UVECTOR(short)
SYCL_ADD_VEC_UVECTOR(long)
SYCL_ADD_VEC_VECTOR(float)
SYCL_ADD_VEC_VECTOR(double)

#undef SYCL_VEC_SCALAR
#undef SYCL_VEC_USCALAR
#undef SYCL_VEC_VECTOR
#undef SYCL_VEC_UVECTOR
#undef SYCL_ADD_VEC_VECTOR
#undef SYCL_ADD_VEC_UVECTOR

#undef SYCL_ENABLE_IF_DIM

} // namespace sycl
} // namespace cl
