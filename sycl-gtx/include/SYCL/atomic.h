#pragma once

// 3.4 Synchronization

#include <atomic>

namespace cl {
namespace sycl {

template <typename T>
class atomic<T> {
 public:
  // Constructors
  atomic() = delete;

  // Methods
  // Only memory_order_relaxed is supported in SYCL 1.2
  void store(T, std::memory_order);
  void store(T, std::memory_order) volatile;
  T load(std::memory_order) const;
  T load(std::memory_order) const volatile;
  T exchange(T, std::memory_order);
  T exchange(T, std::memory_order) volatile;
  T compare_exchange_strong(T*, T, std::memory_order success,
                            std::memory_order fail);
  T compare_exchange_strong(T*, T, std::memory_order success,
                            std::memory_order fail) volatile;
  T fetch_add(T, std::memory_order);
  T fetch_add(T, std::memory_order) volatile;
  T fetch_sub(T, std::memory_order);
  T fetch_sub(T, std::memory_order) volatile;
  T fetch_and(T, std::memory_order);
  T fetch_and(T, std::memory_order) volatile;
  T fetch_or(T, std::memory_order);
  T fetch_or(T, std::memory_order) volatile;
  T fetch_xor(T, std::memory_order);
  T fetch_xor(T, std::memory_order) volatile;

  // Additional functionality provided beyond that of C++11
  T fetch_min(T, std::memory_order);
  T fetch_min(T, std::memory_order) volatile;
  T fetch_max(T, std::memory_order);
  T fetch_max(T, std::memory_order) volatile;
};

typedef atomic<int> atomic_int;
typedef atomic<unsigned int> atomic_uint;
typedef atomic<float> atomic_float;

template <class T>
T atomic_load_explicit(atomic<T>*, std::memory_order);
template <class T>
T atomic_load_explicit(volatile atomic<T>*, std::memory_order);
template <class T>
void atomic_store_explicit(atomic<T>*, T, std::memory_order);
template <class T>
void atomic_store_explicit(volatile atomic<T>*, T, std::memory_order);
template <class T>
T atomic_exchange_explicit(atomic<T>*, T, std::memory_order);
template <class T>
T atomic_exchange_explicit(volatile atomic<T>*, T, std::memory_order);
template <class T>
bool atomic_compare_exchange_strong_explicit(atomic<T>*, T*, T,
                                             std::memory_order success,
                                             std::memory_order fail);
template <class T>
bool atomic_compare_exchange_strong_explicit(volatile atomic<T>*, T*, T,
                                             std::memory_order success,
                                             std::memory_order fail);
template <class T>
T atomic_fetch_add_explicit(atomic<T>*, T, std::memory_order);
template <class T>
T atomic_fetch_add_explicit(volatile atomic<T>*, T, std::memory_order);
template <class T>
T atomic_fetch_sub_explicit(atomic<T>*, T, std::memory_order);
template <class T>
T atomic_fetch_sub_explicit(volatile atomic<T>*, T, std::memory_order);
template <class T>
T atomic_fetch_and_explicit(atomic<T>*, T, std::memory_order);
template <class T>
T atomic_fetch_and_explicit(volatile atomic<T>*, T, std::memory_order);
template <class T>
T atomic_fetch_or_explicit(atomic<T>*, T, std::memory_order);
template <class T>
T atomic_fetch_or_explicit(volatile atomic<T>*, T, std::memory_order);
template <class T>
T atomic_fetch_xor_explicit(atomic<T>*, T, std::memory_order);
template <class T>
T atomic_fetch_xor_explicit(volatile atomic<T>*, T, std::memory_order);

// Additional functionality beyond that provided by C++11
template <class T>
T atomic_fetch_min_explicit(atomic<T>*, T, std::memory_order);
template <class T>
T atomic_fetch_min_explicit(volatile atomic<T>*, T, std::memory_order);
template <class T>
T atomic_fetch_max_explicit(atomic<T>*, T, std::memory_order);
template <class T>
T atomic_fetch_max_explicit(volatile atomic<T>*, T, std::memory_order);

}  // namespace sycl
}  // namespace cl
