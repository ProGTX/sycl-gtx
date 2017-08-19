#pragma once

// 3.7.2 Vector types
// B.5 vec class base

#include "SYCL/detail/common.h"
#include "SYCL/detail/counter.h"
#include "SYCL/detail/data_ref.h"
#include "SYCL/vectors/cl_vec.h"
#include "SYCL/vectors/helpers.h"

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

#define SYCL_ENABLE_IF_DIM(dim) \
  typename std::enable_if<num == dim>::type* = nullptr

/**
 * 3.7.2 Vector types
 *
 * B.5 vec class base
 */
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
    return '_' + type_name() + '_' +
           get_string<counter_t>::get(this->get_count_id());
  }

  string_class this_name() const {
    return type_name() + ' ' + this->name;
  }

 protected:
  base(string_class assign, bool generate_new = false)
      : data_ref(generate_new ? generate_name() : assign) {
    if (generate_new) {
      kernel_add(this_name() + " = " + assign);
    }
  }

 public:
  using element_type = dataT;
  /** Underlying OpenCL type */
  using vector_t = detail::cl_type<dataT, numElements>;

  base() : data_ref(generate_name()) {
    kernel_add(this_name());
  }

  base(const base& copy) : data_ref(copy.name) {}
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
  base(base&& move) noexcept : data_ref(static_cast<data_ref&&>(*this)) {}
  base& operator=(base&& move) noexcept {
    data_ref::operator=(static_cast<data_ref&&>(*this));
    return *this;
  }
  ~base() = default;

  template <int num = numElements>
  base(const data_ref& x, const data_ref& y, SYCL_ENABLE_IF_DIM(2))
      : base(open_parenthesis + type_name() + ")(" + x.name + ", " + y.name +
                 ')',
             true) {}
  template <int num = numElements>
  base(const data_ref& x, const data_ref& y, const data_ref& z,
       SYCL_ENABLE_IF_DIM(3))
      : base(open_parenthesis + type_name() + ")(" + x.name + ", " + y.name +
                 ", " + z.name + ')',
             true) {}
  template <int num = numElements>
  base(const data_ref& x, const data_ref& y, const data_ref& z,
       const data_ref& w, SYCL_ENABLE_IF_DIM(4))
      : base(open_parenthesis + type_name() + ")(" + x.name + ", " + y.name +
                 ", " + z.name + ", " + w.name + ')',
             true) {}
  template <int num = numElements>
  base(const data_ref& s0, const data_ref& s1, const data_ref& s2,
       const data_ref& s3, const data_ref& s4, const data_ref& s5,
       const data_ref& s6, const data_ref& s7, SYCL_ENABLE_IF_DIM(8))
      : base(open_parenthesis + type_name() + ")(" + s0.name + ", " + s1.name +
                 ", " + s2.name + ", " + s3.name + ", " + s4.name + ", " +
                 s5.name + ", " + s6.name + ", " + s7.name + ')',
             true) {}
  template <int num = numElements>
  base(const data_ref& s0, const data_ref& s1, const data_ref& s2,
       const data_ref& s3, const data_ref& s4, const data_ref& s5,
       const data_ref& s6, const data_ref& s7, const data_ref& s8,
       const data_ref& s9, const data_ref& sA, const data_ref& sB,
       const data_ref& sC, const data_ref& sD, const data_ref& sE,
       const data_ref& sF, const data_ref& sG, const data_ref& sH,
       SYCL_ENABLE_IF_DIM(16))
      : base(open_parenthesis + type_name() + ")(" + s0.name + ", " + s1.name +
                 ", " + s2.name + ", " + s3.name + ", " + s4.name + ", " +
                 s5.name + ", " + s6.name + ", " + s7.name + ", " + s8.name +
                 ", " + s9.name + ", " + sA.name + ", " + sB.name + ", " +
                 sC.name + ", " + sD.name + ", " + sE.name + ", " + sF.name +
                 ')',
             true) {}

  operator vec<dataT, numElements>&() {
    return *reinterpret_cast<vec<dataT, numElements>*>(this);  // NOLINT
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

// TODO(progtx): Swizzle methods
// swizzled_vec<T, out_dims> swizzle<int s1, ...>();
#ifdef SYCL_SIMPLE_SWIZZLES
// swizzled_vec<T, 4> xyzw();
//...
#endif  // #ifdef SYCL_SIMPLE_SWIZZLES
};

}  // namespace vectors
}  // namespace detail

}  // namespace sycl
}  // namespace cl
