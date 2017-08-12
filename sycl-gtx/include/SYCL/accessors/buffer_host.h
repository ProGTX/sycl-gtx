#pragma once

// Host buffer accessors
// 3.6.4.7 Host accessors

#include "SYCL/access.h"
#include "SYCL/accessor.h"
#include "SYCL/accessors/buffer_base.h"
#include "SYCL/detail/synchronizer.h"
#include "SYCL/ranges.h"
#include <array>

namespace cl {
namespace sycl {
namespace detail {

#define SYCL_ACCESSOR_HOST_REF_CONSTRUCTOR()                                \
  using acc_t = accessor_detail<DataType, dimensions, mode,                 \
                                access::target::host_buffer>;               \
  friend acc_t;                                                             \
  template <int, typename, int, access::mode>                               \
  friend class accessor_host_ref;                                           \
  const acc_t* parent;                                                      \
  std::array<::size_t, 3> rang;                                             \
  accessor_host_ref(const acc_t* parent, std::array<::size_t, 3> rang)      \
      : parent(parent), rang(rang) {}                                       \
  accessor_host_ref(const acc_t* parent, const accessor_host_ref& copy)     \
      : parent(parent), rang(copy.rang) {}                                  \
  accessor_host_ref(const acc_t* parent, accessor_host_ref&& move) noexcept \
      : parent(parent), rang(std::move(move.rang)) {}                       \
  friend void swap(accessor_host_ref& first, accessor_host_ref& second) {   \
    std::swap(first.rang, second.rang);                                     \
  }

template <int level, typename DataType, int dimensions, access::mode mode>
class accessor_host_ref {
 protected:
  using Lower = accessor_host_ref<dimensions - 1, DataType, dimensions, mode>;
  SYCL_ACCESSOR_HOST_REF_CONSTRUCTOR();

 public:
  Lower operator[](int index) {
    auto rang_copy = rang;
    rang_copy[dimensions - level] = index;
    return Lower(parent, rang_copy);
  }
};

template <typename DataType, int dimensions, access::mode mode>
class accessor_host_ref<1, DataType, dimensions, mode> {
 protected:
  SYCL_ACCESSOR_HOST_REF_CONSTRUCTOR();

 public:
  typename base_host_data<DataType>::type& operator[](int index) {
    // http://stackoverflow.com/questions/7367770
    rang[dimensions - 1] = index;
    index = 0;
    int multiplier = 1;
    for (int i = 0; i < dimensions; ++i) {
      index += static_cast<int>(rang[i] * multiplier);
      multiplier *= static_cast<int>(parent->access_buffer_range(i));
    }
    return parent->access_host_data()[index];
  }
};

SYCL_ACCESSOR_CLASS(target == access::target::host_buffer)
, public accessor_buffer<DataType, dimensions>,
    public accessor_host_ref<dimensions, DataType, dimensions, mode> {
  template <int, typename, int, access::mode>
  friend class accessor_host_ref;

  using base_acc_buffer = accessor_buffer<DataType, dimensions>;
  using base_acc_host_ref =
      accessor_host_ref<dimensions, DataType, dimensions, mode>;

 public:
  accessor_detail(buffer<DataType, dimensions> & bufferRef,
                  range<dimensions> offset, range<dimensions> range)
      : base_acc_buffer(bufferRef, nullptr, offset, range),
        base_acc_host_ref(this, std::array<::size_t, 3>{0, 0, 0}) {
    synchronizer::add(this, base_acc_buffer::buf);
  }
  accessor_detail(buffer<DataType, dimensions> & bufferRef)
      : accessor_detail(bufferRef, detail::empty_range<dimensions>(),
                        bufferRef.get_range()) {
    synchronizer::add(this, base_acc_buffer::buf);
  }
  accessor_detail(const accessor_detail& copy)
      : base_acc_buffer(static_cast<const base_acc_buffer&>(copy)),
        base_acc_host_ref(this, copy) {
    synchronizer::add(this, base_acc_buffer::buf);
  }
  accessor_detail(accessor_detail && move) noexcept
      : base_acc_buffer(std::move(static_cast<base_acc_buffer&&>(move))),
        base_acc_host_ref(this,
                          std::move(static_cast<base_acc_host_ref&&>(move))) {
    synchronizer::add(this, base_acc_buffer::buf);
  }

  accessor_detail& operator=(const accessor_detail& copy) {
    base_acc_buffer::operator=(copy);
    base_acc_host_ref tmp(copy);
    std::swap(static_cast<base_acc_host_ref&>(*this), tmp);
    return *this;
  }
  accessor_detail& operator=(accessor_detail&& move) noexcept {
    base_acc_buffer::operator=(std::move(move));
    std::swap(static_cast<base_acc_host_ref&>(*this), move);
    return *this;
  }

  ~accessor_detail() {
    synchronizer::remove(this, base_acc_buffer::buf);
  }
};

}  // namespace detail
}  // namespace sycl
}  // namespace cl

#undef SYCL_ACCESSOR_HOST_REF_CONSTRUCTOR
