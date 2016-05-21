#pragma once

// 3.7.2 Vector types
// B.5 vec class

#include "helpers.h"
#include "vec_members.h"
#include "cl_vec.h"
#include "../access.h"
#include "../../common.h"
#include "../../counter.h"
#include "../../data_ref.h"


namespace cl {
namespace sycl {

// Forward declarations
template <typename, int>
class vec;

template <typename dataT, int numElements>
using swizzled_vec = vec<dataT, numElements>;

namespace detail {
namespace vectors {

// Forward declaration
template <int, int, int...>
struct swizzled;

#define SYCL_ENABLE_IF_DIM(dim)  \
typename std::enable_if<num == dim>::type* = nullptr


template <typename dataT, int numElements>
class base : protected counter<base<dataT, numElements>>, public data_ref {
private:
  template <typename>
  friend struct ::cl::sycl::detail::type_string;
  template <typename, int>
  friend class vec;

  static const int half_size = (numElements + 1) / 2;

  static string_class type_name() {
    return cl_base<dataT, numElements, 0>::type_name();
  }

  string_class generate_name() const {
    return '_' + type_name() + '_' + get_string<counter_t>::get(this->get_count_id());
  }

  string_class this_name() const {
    return type_name() + ' ' + this->name;
  }

protected:
  base(string_class assign, bool generate_new = false)
    : data_ref(generate_new ? generate_name() : assign) {
    if(generate_new) {
      kernel_add(this_name() + " = " + assign);
    }
  }

public:
  using element_type = dataT;
  // Underlying OpenCL type
  using vector_t = detail::cl_type<dataT, numElements>;

  base()
    : data_ref(generate_name()) {
    kernel_add(this_name());
  }

  base(const base& copy)
    : data_ref(copy.name) {}
  base& operator=(const base& copy) {
    data_ref::operator=(copy.name);
    return *this;
  }
  base& operator=(const data_ref& copy) {
    data_ref::operator=(copy.name);
    return *this;
  }
  base& operator=(const dataT& n) {
    data_ref::operator=(n);
    return *this;
  }

  template <int num = numElements>
  base(const data_ref& x, const data_ref& y, SYCL_ENABLE_IF_DIM(2))
    : base(
        open_parenthesis + type_name() + ")(" + x.name + ", " + y.name + ')',
        true
    ) {}

  template <int num = numElements>
  base(const data_ref& x, const data_ref& y, const data_ref& z, SYCL_ENABLE_IF_DIM(3))
    : base(
        open_parenthesis + type_name() + ")(" +
          x.name + ", " + y.name + ", " + z.name +
        ')',
        true
    ) {}

  operator vec<dataT, numElements>&() {
    return *reinterpret_cast<vec<dataT, numElements>*>(this);
  }

  ::size_t get_count() const {
    return numElements;
  }
  ::size_t get_size() const {
    return numElements * sizeof(typename cl_type<dataT, numElements>::type);
  }

  template <int... indices>
  swizzled_vec<dataT, sizeof...(indices)> swizzle() const {
    static const auto size = sizeof...(indices);
    static_assert(size > 0, "Cannot swizzle to zero elements");

    // One extra for final null char
    char access_name[size + 1];
    swizzled<0, indices...>::get(access_name);
    access_name[size] = 0;

    return swizzled_vec<dataT, size>(this->name + ".s" + access_name);
  }

  swizzled_vec<dataT, half_size> lo() const {
    return swizzled_vec<dataT, half_size>(this->name + ".lo");
  }
  swizzled_vec<dataT, half_size> hi() const {
    return swizzled_vec<dataT, half_size>(this->name + ".hi");
  }

  // TODO: Swizzle methods
  //swizzled_vec<T, out_dims> swizzle<int s1, ...>();
#ifdef SYCL_SIMPLE_SWIZZLES
  //swizzled_vec<T, 4> xyzw();
  //...
#endif // #ifdef SYCL_SIMPLE_SWIZZLES
};

} // namespace vectors
} // namespace detail


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

  template <typename T>
  vec& assign(const T& copy) {
    if(this->type == type_t::expression) {
      Base b(this->name, true);
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

  template <typename T>
  vec& assign(const T& copy) {
    if(this->type == type_t::expression) {
      Base b(this->name, true);
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
