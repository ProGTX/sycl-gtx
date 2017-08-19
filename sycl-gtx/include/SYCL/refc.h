#pragma once

// 2.5.8 Managing object lifetimes
// "All OpenCL objects encapsulated in SYCL objects will be reference-counted
// and destroyed once all references have been released."

#include "SYCL/detail/common.h"
#include "SYCL/error_handler.h"

namespace cl {
namespace sycl {

namespace detail {

template <class CL_Type>
using cl_resource_f = ::cl_int(CL_API_CALL*)(CL_Type);

template <class CL_Type>
::cl_int CL_API_CALL cl_do_nothing(CL_Type) {
  return CL_SUCCESS;
}

template <class CL_Type>
using refc_ptr = shared_ptr_class<typename std::remove_pointer<CL_Type>::type>;

template <class CL_Type,
          cl_resource_f<CL_Type> retain = &cl_do_nothing<CL_Type>,
          cl_resource_f<CL_Type> release = &cl_do_nothing<CL_Type> >
class refc : public refc_ptr<CL_Type> {
 private:
  using Base = refc_ptr<CL_Type>;

 public:
  static void call_release(CL_Type data) {
    auto error_code = release(data);
    error::report(error_code);
  }

  static void call_retain(CL_Type data) {
    if (data != nullptr) {
      auto error_code = retain(data);
      error::report(error_code);
    }
  }

  refc() : Base(nullptr, release) {}

  refc(CL_Type data) : Base(data, release) {
    call_retain(data);
  }

  refc(const refc&) = default;
  refc(refc&& move) noexcept : Base(std::move(move)) {}
  refc& operator=(const refc&) = default;
  refc& operator=(refc&& move) noexcept {
    Base::operator=(std::move(move));
    return *this;
  }
  ~refc() = default;

  void reset(CL_Type data) {
    Base::reset(data, release);
    call_retain(data);
  }

  void release_one() {
    call_release(this->get());
  }

  refc& operator=(CL_Type data) {
    reset(data);
    return *this;
  }
};

}  // namespace detail

}  // namespace sycl
}  // namespace cl
