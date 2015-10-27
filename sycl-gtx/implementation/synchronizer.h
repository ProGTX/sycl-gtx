#pragma once

#include "common.h"
#include <set>
#include <unordered_map>

namespace cl {
namespace sycl {

// Forward declaration
class queue;

namespace detail {

// Forward declarations
class accessor_base;
class buffer_base;

class synchronizer {
private:
	static std::set<queue*> queues;
	static std::unordered_map<accessor_base*, buffer_base*> host_accessors;

	static void wait_on_queues(buffer_base* buf);
	static void flush_queues(buffer_base* buf);

public:
	static void add(queue* q);
	static void remove(queue* q);
	static void add(accessor_base* acc, buffer_base* buf);
	static void remove(accessor_base* acc, buffer_base* buf);

	static bool can_flush(const std::set<detail::buffer_base*>& buffers_in_use);
};

} // namespace detail

} // namespace sycl
} // namespace cl
