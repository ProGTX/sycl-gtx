#pragma once

// 3.6.2 Exception Class Interface

#include "SYCL/detail/common.h"
#include "SYCL/detail/error_code.h"
#include <exception>

namespace cl {
namespace sycl {

// Forward declarations
class context;
namespace detail {
namespace error {
struct thrower;
}
}  // namespace detail

struct exception : std::exception_ptr {
 protected:
  friend struct detail::error::thrower;
  string_class description;
  context* thrower;

  exception(string_class description, context* thrower = nullptr)
      : description(description), thrower(thrower) {}

 public:
  exception() : exception("Undefined SYCL Error") {}

  /** Returns a descriptive string for the error, if available. */
  string_class what() const {
    return description;
  }

  /**
   * Returns the context that caused the error.
   * Returns null if not a buffer error.
   */
  context* get_context() {
    return thrower;
  }
};

struct cl_exception : exception {
 private:
  friend struct detail::error::thrower;
  ::cl_int error_code;

  cl_exception(::cl_int error_code, context* thrower = nullptr)
      : exception(detail::error_string(error_code), thrower),
        error_code(error_code) {}

 public:
  cl_exception() : cl_exception(CL_SUCCESS, nullptr) {}

  // Thrown as a result of an OpenCL API error code
  ::cl_int get_cl_code() const {
    return error_code;
  }
};

struct async_exception : exception {
  // stored in an exception_list for asynchronous errors
};

using exception_ptr = std::exception_ptr;

// TODO(progtx): Used as a container for a list of asynchronous exceptions
class exception_list {
 private:
  using list_t = vector_class<async_exception>;
  list_t list;

 public:
  using value_type = exception_ptr;
  using reference = value_type&;
  using const_reference = const value_type&;
  using size_type = ::size_t;
  using iterator = list_t::const_iterator;  // TODO(progtx): non const
  using const_iterator = list_t::const_iterator;

  ::size_t size() const {
    return list.size();
  }

  /** first asynchronous exception */
  iterator begin() const {
    return list.begin();
  }
  /** refer to past-the-end last asynchronous exception */
  iterator end() const {
    return list.end();
  }
};

using async_handler = function_class<void(cl::sycl::exception_list)>;

// TODO(progtx):
class runtime_error : public exception {};
class kernel_error : public runtime_error {};
class accessor_error : public runtime_error {};
class nd_range_error : public runtime_error {};
class event_error : public runtime_error {};
class invalid_parameter_error : public runtime_error {};
class device_error : public exception {};
class compile_program_error : public device_error {};
class link_program_error : public device_error {};
class invalid_object_error : public device_error {};
class memory_allocation_error : public device_error {};
class platform_error : public device_error {};
class profiling_error : public device_error {};
class feature_not_supported : public device_error {};

}  // namespace sycl
}  // namespace cl
