#pragma once

#include "SYCL/access.h"
#include "SYCL/accessor.h"
#include "SYCL/detail/common.h"
#include "SYCL/detail/data_ref.h"
#include "SYCL/detail/src_handlers/register_resource.h"
#include "SYCL/ranges/id.h"

namespace cl {
namespace sycl {
namespace detail {

// Forward declaration
template <int level, typename DataType, int dimensions, access::mode mode,
          access::target target>
class accessor_device_ref;

template <typename DataType>
struct acc_device_return {
  using type = data_ref;
};

template <int level, typename DataType, int dimensions, access::mode mode,
          access::target target>
struct subscript_helper {
  using type =
      accessor_device_ref<level - 1, DataType, dimensions, mode, target>;
};
template <typename DataType, int dimensions, access::mode mode,
          access::target target>
struct subscript_helper<1, DataType, dimensions, mode, target> {
  using type = typename acc_device_return<DataType>::type;
};

#define SYCL_ACCESSOR_DEVICE_REF_CONSTRUCTOR()                                \
  using acc_t = accessor_detail<DataType, dimensions, mode, target>;          \
  friend acc_t;                                                               \
  template <int, typename, int, access::mode, access::target>                 \
  friend class accessor_device_ref;                                           \
  const acc_t* parent;                                                        \
  vector_class<string_class> rang;                                            \
  accessor_device_ref(const acc_t* parent, vector_class<string_class> range)  \
      : parent(parent), rang(range) {                                         \
    rang.resize(3);                                                           \
  }                                                                           \
  accessor_device_ref(const acc_t* parent, const accessor_device_ref& copy)   \
      : parent(parent), rang(copy.rang) {}                                    \
  accessor_device_ref(const acc_t* parent,                                    \
                      accessor_device_ref&& move) noexcept                    \
      : parent(parent), rang(std::move(move.rang)) {}                         \
  friend void swap(accessor_device_ref& first, accessor_device_ref& second) { \
    std::swap(first.rang, second.rang);                                       \
  }

#define SYCL_DEVICE_REF_SUBSCRIPT_OP(prefix, type)         \
  subscript_return_t operator[](const type& index) const { \
    return prefix subscript(index);                        \
  }

#define SYCL_DEVICE_REF_SUBSCRIPT_OPERATORS(prefix) \
  SYCL_DEVICE_REF_SUBSCRIPT_OP(prefix, data_ref);   \
  SYCL_DEVICE_REF_SUBSCRIPT_OP(prefix, ::size_t);

template <int level, typename DataType, int dimensions, access::mode mode,
          access::target target>
class accessor_device_ref {
 protected:
  using subscript_return_t =
      typename subscript_helper<dimensions, DataType, dimensions, mode,
                                target>::type;
  SYCL_ACCESSOR_DEVICE_REF_CONSTRUCTOR();
  template <class T>
  subscript_return_t subscript(const T& index) const {
    auto rang_copy = rang;
    rang_copy[dimensions - level] = data_ref::get_name(index);
    return subscript_return_t(parent, rang_copy);
  }

 public:
  SYCL_DEVICE_REF_SUBSCRIPT_OPERATORS(this->);
};

template <typename DataType, int dimensions, access::mode mode,
          access::target target>
class accessor_device_ref<1, DataType, dimensions, mode, target> {
 protected:
  using subscript_return_t = typename acc_device_return<DataType>::type;
  SYCL_ACCESSOR_DEVICE_REF_CONSTRUCTOR();

  template <class T>
  subscript_return_t subscript(const T& index) const {
    // Basically the same as with host buffer accessor, just dealing with
    // strings
    auto rang_copy = rang;
    rang_copy[dimensions - 1] = data_ref::get_name(index);
    string_class ind(std::move(rang_copy[0]));
    auto multiplier = parent->access_buffer_range(0);
    for (int i = 1; i < dimensions; ++i) {
      ind += string_class(" + ") + std::move(rang_copy[i]) + " * " +
             get_string<decltype(multiplier)>::get(multiplier);
      multiplier *= parent->access_buffer_range(i);
    }
    auto resource_name = kernel_ns::register_resource(*parent);
    return subscript_return_t(resource_name + "[" + ind + "]");
  }

 public:
  SYCL_DEVICE_REF_SUBSCRIPT_OPERATORS(this->);
};

}  // namespace detail
}  // namespace sycl
}  // namespace cl

#undef SYCL_ACCESSOR_DEVICE_REF_CONSTRUCTOR
