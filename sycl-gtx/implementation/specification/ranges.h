#pragma once

// 3.4.1 Ranges and identifiers

#include "../common.h"
#include "../debug.h"

namespace cl {
namespace sycl {

namespace detail {

// Forward declaration
template <typename DataType, int dimensions>
class buffer;

template <int dimensions>
struct range_ {
protected:
	int dims[3];

	range_(size_t first, size_t second, size_t third)
#if MSVC_LOW
	{
		dims[0] = first;
		dims[1] = second;
		dims[2] = third;
#else
		: dims{ first, second, third } {
#endif
	}

	range_(const range_&) = default;

public:
	int operator[](size_t n) {
		return dims[n];
	}
};

} // namespace detail

template <int dimensions = 1>
class range;

template <>
struct range<1> : detail::range_<1> {
	range(size_t size)
		: detail::range_<1>(size, 1, 1) {}
	range(size_t size[1])
		: range(size[0]) {}
};
template <>
struct range<2> : detail::range_<2>{
	range(size_t sizeX, size_t sizeY)
		: detail::range_<2>(sizeX, sizeY, 1) {}
	range(size_t size[2])
		: range(size[0], size[1]) {}
};
template <>
struct range<3> : detail::range_<3>{
	range(size_t sizeX, size_t sizeY, size_t sizeZ)
		: detail::range_<3>(sizeX, sizeY, sizeZ) {}
	range(size_t size[3])
		: range(size[0], size[1], size[2]) {}
};

template <int dimensions = 1>
class id {
private:
	range<dimensions> global_size;
	range<dimensions> local_size;
public:
	id(
		range<dimensions> global_size = range<dimensions>(VECTOR_CLASS<size_t>(dimensions, 0).data()),
		range<dimensions> local_size = range<dimensions>(VECTOR_CLASS<size_t>(dimensions, 0).data())
	)	: global_size(global_size), local_size(local_size) {}
	id(int n)
		: id() {
		// TODO: Not sure if this is correct
		global_size[0] = n;
		local_size[0] = n;
	}
	int get(int dimension);
};

template <int dimensions = 1>
class nd_range {
private:
	range<dimensions> global_size;
	range<dimensions> local_size;

	// TODO
	id<dimensions> offset;

public:
	nd_range(range<dimensions> global_size, range<dimensions> local_size, id<dimensions> offset = id<dimensions>())
		: global_size(global_size), local_size(local_size), offset(offset) {}

	range<dimensions> get_global_range() {
		return global_size;
	}
	range<dimensions> get_local_range() {
		return local_size;
	}
	range<dimensions> get_group_range();
};

// TODO: class item {};


} // namespace sycl
} // namespace cl
