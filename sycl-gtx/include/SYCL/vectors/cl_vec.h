#pragma once

#include "SYCL/detail/common.h"
#include "SYCL/vectors/cl_type.h"
#include <initializer_list>

namespace cl {
namespace sycl {

namespace detail {
namespace vectors {

// Forward declaration
template <typename, int>
class base;

/** In OpenCL, 3 and 4 element vectors have the same size */
template <int parentElems>
struct num_elems {
  static const int value = parentElems;
};
template <>
struct num_elems<3> {
  static const int value = 4;
};

template <typename dataT, int parentElems, int selfElems = parentElems>
struct cl_base;

template <typename dataT>
struct cl_base<dataT, 1, 1> {
 private:
  dataT elem;

 public:
  cl_base() {}
  cl_base(dataT value) : elem(value) {}

  operator dataT&() {
    return elem;
  }
  operator const dataT&() const {
    return elem;
  }
};

template <typename dataT, int parentElems>
struct cl_base<dataT, parentElems, 0> {
 protected:
  template <typename, int, int>
  friend struct cl_base;
  template <typename>
  friend struct ::cl::sycl::detail::type_string;
  template <typename, int>
  friend class base;

  using genvector = typename detail::cl_type<dataT, parentElems>::type;

  static const int size = num_elems<parentElems>::value;
  dataT elems[size];

  static string_class type_name() {
    return type_string<dataT>::get() +
           ((parentElems == 1) ? "" : get_string<int>::get(parentElems));
  }

 public:
  cl_base() {}
  cl_base(const cl_base&) = default;
  cl_base(genvector v) {
    auto start = reinterpret_cast<dataT*>(&v);  // NOLINT
    std::copy(start, start + size, this->elems);
  }
#if MSVC_2013_OR_LOWER
  cl_base(cl_base&& move) {
    std::swap(this->elems, move.elems);
  }
  cl_base& operator=(cl_base&& move) {
    std::swap(this->elems, move.elems);
    return *this;
  }
#elif MSVC_2017_OR_LOWER
  cl_base(cl_base&&) = default;
  cl_base& operator=(cl_base&&) = default;
#else
  cl_base(cl_base&&) noexcept = default;             // NOLINT
  cl_base& operator=(cl_base&&) noexcept = default;  // NOLINT
#endif
  cl_base& operator=(const cl_base&) = default;
  cl_base& operator=(genvector v) {
    std::copy(&v, &v + size, this->elems);
  }

  ~cl_base() = default;

  operator genvector&() {
    return *reinterpret_cast<genvector*>(elems);  // NOLINT
  }
  operator const genvector&() const {
    return *reinterpret_cast<genvector*>(elems);  // NOLINT
  }
};

#define SYCL_CL_VEC_INHERIT_CONSTRUCTORS \
  cl_base() {}                           \
  cl_base(const cl_base&) = default;     \
  cl_base(genvector v) : Base(v) {}      \
  cl_base(cl_base&& move) {              \
    std::swap(this->elems, move.elems);  \
  }

#define SYCL_CL_REF(return_t, name, code) \
  return_t& name() {                      \
    return code;                          \
  }                                       \
  const return_t& name() const {          \
    return code;                          \
  }

template <typename dataT, int parentElems>
struct cl_base<dataT, parentElems, 2> : cl_base<dataT, parentElems, 0> {
 protected:
  using Base = cl_base<dataT, parentElems, 0>;
  using cl_base_2 = cl_base<dataT, 2, 2>;

 public:
#if MSVC_2013_OR_LOWER
  SYCL_CL_VEC_INHERIT_CONSTRUCTORS
#else
  using Base::Base;
#endif

  SYCL_CL_REF(dataT, x, this->elems[0]);
  SYCL_CL_REF(dataT, y, this->elems[1]);
  SYCL_CL_REF(dataT, lo, x());
  SYCL_CL_REF(dataT, hi, y());
  SYCL_CL_REF(cl_base_2, xy, *reinterpret_cast<cl_base_2*>(this));  // NOLINT
};

template <typename dataT, int parentElems>
struct cl_base<dataT, parentElems, 3> : cl_base<dataT, parentElems, 2> {
 protected:
  using Base = cl_base<dataT, parentElems, 2>;
  using cl_base_2 = typename Base::cl_base_2;
  using cl_base_3 = cl_base<dataT, 3, 3>;

 public:
#if MSVC_2013_OR_LOWER
  SYCL_CL_VEC_INHERIT_CONSTRUCTORS
#else
  using Base::Base;
#endif

  SYCL_CL_REF(dataT, z, this->elems[2]);
  SYCL_CL_REF(cl_base_2, lo, this->xy());
  SYCL_CL_REF(dataT, hi, z());
  SYCL_CL_REF(cl_base_3, xyz, *reinterpret_cast<cl_base_3*>(this));  // NOLINT
};

#define SYCL_CL_HI(cl_base_half, half)              \
  cl_base_half hi() const {                         \
    cl_base_half ret;                               \
    using type = decltype(this->elems + 0);         \
    reinterpret_cast<type&>(ret.elems) =            \
        reinterpret_cast<type>(this->elems + half); \
    return ret;                                     \
  }

template <typename dataT, int parentElems>
struct cl_base<dataT, parentElems, 4> : cl_base<dataT, parentElems, 3> {
 private:
  using Base = cl_base<dataT, parentElems, 3>;
  using cl_base_2 = typename Base::cl_base_2;
  using cl_base_3 = typename Base::cl_base_3;
  using cl_base_4 = cl_base<dataT, 4, 4>;

 public:
#if MSVC_2013_OR_LOWER
  SYCL_CL_VEC_INHERIT_CONSTRUCTORS
#else
  using Base::Base;
#endif

  SYCL_CL_REF(dataT, w, this->elems[3]);
  SYCL_CL_REF(cl_base_4, xyzw, *reinterpret_cast<cl_base_4*>(this));  // NOLINT
  SYCL_CL_REF(cl_base_2, hi,
              *reinterpret_cast<cl_base_2*>(                    // NOLINT
                  reinterpret_cast<dataT*>(this->elems) + 2));  // NOLINT

  operator cl_base_3&() {
    return this->xyz();
  }
  operator const cl_base_3&() const {
    return this->xyz();
  }
};

template <typename dataT, int parentElems>
struct cl_base<dataT, parentElems, 8> : cl_base<dataT, parentElems, 0> {
 private:
  using Base = cl_base<dataT, parentElems, 0>;
  using cl_base_4 = cl_base<dataT, 4, 4>;

 public:
#if MSVC_2013_OR_LOWER
  SYCL_CL_VEC_INHERIT_CONSTRUCTORS
#else
  using Base::Base;
#endif

  SYCL_CL_REF(cl_base_4, lo, *reinterpret_cast<cl_base_4*>(this));  // NOLINT
  SYCL_CL_REF(cl_base_4, hi,
              *reinterpret_cast<cl_base_4*>(                    // NOLINT
                  reinterpret_cast<dataT*>(this->elems) + 4));  // NOLINT
};

template <typename dataT, int parentElems>
struct cl_base<dataT, parentElems, 16> : cl_base<dataT, parentElems, 0> {
 private:
  using Base = cl_base<dataT, parentElems, 0>;
  using cl_base_8 = cl_base<dataT, 8, 8>;

 public:
#if MSVC_2013_OR_LOWER
  SYCL_CL_VEC_INHERIT_CONSTRUCTORS
#else
  using Base::Base;
#endif

  SYCL_CL_REF(cl_base_8, lo, *reinterpret_cast<cl_base_8*>(this));  // NOLINT
  SYCL_CL_REF(cl_base_8, hi,
              *reinterpret_cast<cl_base_8*>(                    // NOLINT
                  reinterpret_cast<dataT*>(this->elems) + 8));  // NOLINT
};

#undef SYCL_CL_REF
#undef SYCL_CL_VEC_INHERIT_CONSTRUCTORS

}  // namespace vectors
}  // namespace detail

}  // namespace sycl
}  // namespace cl
