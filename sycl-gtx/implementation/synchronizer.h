#pragma once

#include "common.h"
#include <set>

namespace cl {
namespace sycl {

// Forward declaration
class queue;

namespace detail {

// Forward declaration
class buffer_base;

class synchronizer {
private:
	static std::set<queue*> queues;

public:
	static void add(queue* q);
	static void remove(queue* q);
	static void barrier(buffer_base* buf);
};

} // namespace detail

} // namespace sycl
} // namespace cl
