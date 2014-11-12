#pragma once

// 3.3.1 Buffers

#include "../common.h"
#include "../debug.h"
#include "access.h"
#include "ranges.h"
#include <vector>

namespace cl {
namespace sycl {

// Forward declaration.
template <typename dataType, int dimensions, access::mode mode, access::target target>
class accessor;

namespace detail {

template <typename T, int dimensions>
struct buffer {
	buffer(T* host_data, range<dimensions> range) {
		DSELF() << "not implemented";
	}
	buffer(T* host_data, int range) {
		DSELF() << "not implemented";
	}

	range<dimensions> get_range();
	size_t get_count();
	size_t get_size();

	template<access::mode mode, access::target target = access::global_buffer>
	accessor<T, dimensions, mode, target> get_access() {
		DSELF() << "not implemented";
	}
};

} // namespace detail

// typename T: The type of the elements of the buffer
// int dimensions: number of dimensions of the buffer : 1, 2, or 3
template <typename T, int dimensions = 1>
struct buffer : detail::buffer<T, dimensions> {
#if MSVC_LOW
	buffer(T* host_data, range<dimensions> range)
		: detail::buffer<T, dimensions>(host_data, range) {}
	buffer(T* host_data, int range)
		: detail::buffer<T, dimensions>(host_data, range) {}
#else
	using detail::buffer<T, dimensions>::buffer;
#endif
};

template <typename T>
struct buffer<T, 1> : detail::buffer<T, 1> {
#if MSVC_LOW
	buffer(T* host_data, range<1> range)
		: detail::buffer<T, 1>(host_data, range) {}
	buffer(T* host_data, int range)
		: detail::buffer<T, 1>(host_data, range) {}
#else
	using detail::buffer<T, 1>::buffer;
#endif
	buffer(std::vector<T> host_data) : detail::buffer<T, 1>(host_data.data(), host_data.size()) {
		DSELF() << "not implemented";
	}
};

} // namespace sycl
} // namespace cl
