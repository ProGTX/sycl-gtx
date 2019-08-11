#pragma once

#include "SYCL/detail/msvc_version.h"
#include "SYCL/detail/opencl.h"

#include <cstdio>
#include <sstream>

#define SYCL_SWAP(member) swap(first.member, second.member)
#define SYCL_MOVE_INIT(member) member(std::move(move.member))

#if MSVC_2013_OR_LOWER
#define SYCL_THREAD_LOCAL __declspec(thread)
#else
#define SYCL_THREAD_LOCAL thread_local
#endif

// 3.2 C++ Standard library classes required for the interface

#ifndef CL_SYCL_NO_STD_VECTOR
#include <vector>
#endif
#ifndef CL_SYCL_NO_STD_STRING
#include <string>
#endif
#ifndef CL_SYCL_NO_STD_FUNCTION
#include <functional>
#endif
#ifndef CL_SYCL_NO_STD_MUTEX
#include <mutex>
#endif
#ifndef CL_SYCL_NO_UNIQUE_PTR
#include <memory>
#endif
#ifndef CL_SYCL_NO_SHARED_PTR
#include <memory>
#endif
#ifndef CL_SYCL_NO_WEAK_PTR
#include <memory>
#endif

namespace cl {
namespace sycl {

#ifndef CL_SYCL_NO_STD_VECTOR
template <class T, class Alloc = ::std::allocator<T>>
using vector_class = ::std::vector<T, Alloc>;
#endif

#ifndef CL_SYCL_NO_STD_STRING
using string_class = ::std::string;
#endif

#ifndef CL_SYCL_NO_STD_FUNCTION
#if MSVC_2013_OR_LOWER
template <class T>
class function_class : public ::std::function<T> {
 private:
  using Base = ::std::function<T>;

 public:
  function_class() {}
  function_class(std::nullptr_t fn) : Base(fn) {}
  template <class Fn>
  function_class(Fn fn) : Base(fn) {}
  function_class(const Base& x) : Base(x) {}
  function_class(Base&& x) : Base(std::move(x)) {}
  function_class(const function_class&) = default;
  function_class& operator=(const function_class&) = default;
  function_class(function_class&& move) : Base(std::move((Base)move)) {}
  function_class& operator=(function_class&& move) {
    Base::operator=(std::move((Base)move));
    return *this;
  }
};
#else
template <class T>
using function_class = ::std::function<T>;
#endif
#endif

#ifndef CL_SYCL_NO_STD_MUTEX
using mutex_class = ::std::mutex;
#endif
#ifndef CL_SYCL_NO_UNIQUE_PTR
template <class T, class D = ::std::default_delete<T>>
using unique_ptr_class = ::std::unique_ptr<T, D>;
#endif
#ifndef CL_SYCL_NO_SHARED_PTR
template <class T>
using shared_ptr_class = ::std::shared_ptr<T>;
#endif
#ifndef CL_SYCL_NO_WEAK_PTR
template <class T>
using weak_ptr_class = ::std::weak_ptr<T>;
#endif

namespace detail {

// http://stackoverflow.com/a/3418285
static bool string_replace_one(string_class& str, const string_class& from,
                               const string_class& to) {
  ::size_t start_pos = str.find(from);
  if (start_pos == string_class::npos) {
    return false;
  }
  str.replace(start_pos, from.length(), to);
  return true;
}

template <class To, class From>
vector_class<To> transform_vector(vector_class<From> array) {
  return vector_class<To>(array.data(), array.data() + array.size());
}

template <class From>
auto get_cl_array(vector_class<From> array)
    -> vector_class<decltype(array[0].get())> {
  vector_class<decltype(array[0].get())> transformed;
  transformed.reserve(array.size());
  for (auto& e : array) {
    transformed.push_back(e.get());
  }
  return transformed;
}

template <typename EnumClass, EnumClass Value, class T>
bool has_extension(T* sycl_class, const string_class& extension_name) {
  // TODO(progtx): Maybe add caching
  return false;  // TODO(progtx): Doesn't seem to work, ignore for now
  // return sycl_class->get_info<Value>().find(extension_name) !=
  // string_class::npos;
}

template <typename DataType>
struct type_string {
  static string_class get() {
    return DataType::type_name();
  }
};

#define SYCL_GET_TYPE_STRING(type) \
  template <>                      \
  struct type_string<type> {       \
    static string_class get() {    \
      return #type;                \
    }                              \
  };

#define SYCL_GET_UTYPE_STRING(type)   \
  SYCL_GET_TYPE_STRING(type)          \
  template <>                         \
  struct type_string<unsigned type> { \
    static string_class get() {       \
      return "u" #type;               \
    }                                 \
  };

SYCL_GET_TYPE_STRING(bool)
SYCL_GET_UTYPE_STRING(int)
SYCL_GET_UTYPE_STRING(char)
SYCL_GET_UTYPE_STRING(short)
SYCL_GET_UTYPE_STRING(long)
SYCL_GET_TYPE_STRING(float)
SYCL_GET_TYPE_STRING(double)

#undef SYCL_GET_TYPE_STRING
#undef SYCL_GET_UTYPE_STRING

template <typename T>
struct data_size {
  static ::size_t get() {
    return sizeof(T);
  }
};

template <typename DataType>
struct base_host_data {
  using type = DataType;
};

template <typename T>
struct get_string {
  static string_class get(const T& t) {
    std::stringstream s;
    s << t;
    return s.str();
  }
};
// TODO(progtx): Should be more efficient
template <>
struct get_string<float> {
  static string_class get(float t) {
    std::stringstream s;
    s << t;
    auto str = s.str();
    if (str.find('e') == string_class::npos &&
        str.find('.') == string_class::npos) {
      str += ".f";
    } else {
      str += 'f';
    }
    return str;
  }
};

template <typename dataT, int numElements>
struct cl_type;

}  // namespace detail

}  // namespace sycl
}  // namespace cl
