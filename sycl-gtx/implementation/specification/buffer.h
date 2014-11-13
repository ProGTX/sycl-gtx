#pragma once

// 3.3.1 Buffers

#include "access.h"
#include "event.h"
#include "ranges.h"
#include "queue.h"
#include "../common.h"
#include "../debug.h"
#include <vector>

namespace cl {
namespace sycl {

// Forward declaration.
template <typename dataType, int dimensions, access::mode mode, access::target target>
class accessor;

namespace detail {

template <typename T, int dimensions>
struct buffer {
	buffer(range<dimensions> range);
	buffer(T* host_data, range<dimensions> range) {
		DSELF() << "not implemented";
	}
	//buffer(storage<T> &store, range<dimensions>);
	//buffer(buffer, index<dimensions> base_index, range<dimensions> sub_range);
	buffer(cl_mem mem_object, queue from_queue, event available_event);

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
	buffer(range<dimensions> range)
		: detail::buffer<T, dimensions>(range) {}
	buffer(T* host_data, range<dimensions> range)
		: detail::buffer<T, dimensions>(host_data, range) {}
	//buffer(storage<T> &store, range<dimensions>);
	//buffer(buffer, index<dimensions> base_index, range<dimensions> sub_range);
	buffer(cl_mem mem_object, queue from_queue, event available_event)
		: detail::buffer<T, dimensions>(mem_object, from_queue, available_event) {}
#else
	using detail::buffer<T, dimensions>::buffer;
#endif
};

template <typename T>
struct buffer<T, 1> : detail::buffer<T, 1> {
#if MSVC_LOW
	buffer(range<1> range)
		: detail::buffer<T, 1>(range) {}
	buffer(T* host_data, range<1> range)
		: detail::buffer<T, 1>(host_data, range) {}
	//buffer(storage<T> &store, range<1>);
	buffer(T* startIterator, T* endIterator);
	//buffer(buffer, index<dimensions> base_index, range<dimensions> sub_range);
	buffer(cl_mem mem_object, queue from_queue, event available_event)
		: detail::buffer<T, 1>(mem_object, from_queue, available_event) {}
#else
	using detail::buffer<T, 1>::buffer;
#endif

	// TODO: Used by the Codeplay SYCL example
	buffer(VECTOR_CLASS<T> host_data)
		: detail::buffer<T, 1>(host_data.data(), host_data.size()) {
		DSELF() << "not implemented";
	}
};

} // namespace sycl
} // namespace cl
