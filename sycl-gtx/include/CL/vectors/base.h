#pragma once

// 3.7.2 Vector types
// B.5 vec class base

#include "helpers.h"
#include "cl_vec.h"
#include "detail/common.h"
#include "detail/counter.h"
#include "detail/data_ref.h"


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

} // namespace sycl
} // namespace cl
