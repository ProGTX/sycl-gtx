#pragma once

#include "SYCL/detail/msvc_version.h"

#include <cmath>
#include <iostream>
#include <sstream>

#if MSVC_2013_OR_LOWER
#define __func__ __FUNCTION__
#define constexpr const
#endif

#define DSELF() debug(__func__)

#ifndef NDEBUG
#define SYCL_ENABLE_DEBUG 1
#endif

#if SYCL_ENABLE_DEBUG
class debug {
 protected:
  static constexpr bool DEBUG_ACTIVE = true;
  std::stringstream stream;
  std::stringstream before;

  template <typename T>
  void AddToStream(T add) {
    if (DEBUG_ACTIVE) {
      stream << add << ' ';
    }
  }

  template <class T>
  void AddToStream(std::basic_string<T> string) {
    if (DEBUG_ACTIVE) {
      stream << string << ' ';
    }
  }

  template <template <class, class...> class Container, class First,
            class... Others>
  void AddToStream(Container<First, Others...> list) {
    for (auto&& element : list) {
      AddToStream(element);
    }
  }

 public:
  debug() = default;
#if MSVC_2013_OR_LOWER
  debug(debug&& move)
      : stream(std::move(move.stream)), before(std::move(move.before)) {}
#elif defined(__GNUC__) && (__GNUC__ < 5)
  // https://gcc.gnu.org/bugzilla/show_bug.cgi?id=54316
  debug(debug&& move) noexcept {
    stream << move.stream.str();
    before << move.before.str();
  }
#else
  debug(debug&& move) = default;
#endif
  debug(const debug& copy) = delete;

  debug& operator=(const debug& copy) = delete;
  debug& operator=(debug&& move) = default;

  template <typename T>
  debug(T add) : debug() {
    before << "Debug: ";
    AddToStream(add);
  }

  template <typename U, typename T>
  debug(U before, T add) : debug() {
    this->before << before;
    AddToStream(add);
  }

  template <typename T>
  debug& operator<<(T add) {
    AddToStream(add);
    return *this;
  }

  ~debug() {
    if (DEBUG_ACTIVE) {
      std::cout << before.str() << stream.str() << std::endl;
    }
  }

  template <typename T>
  static debug warning(T message) {
    return debug("SYCL warning: ", message);
  }
};
#else
class debug {
 public:
  debug() {}

  template <typename T>
  debug(T) {}

  template <typename U, typename T>
  debug(U before, T add) {}

  template <typename T>
  debug& operator<<(T add) {
    return *this;
  }

  template <typename T>
  static debug warning(T message) {
    return debug();
  }

  ~debug() {}
};
#endif
