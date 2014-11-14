#pragma once

// 3.4.1 Ranges and identifiers

#include "../debug.h"

namespace cl {
namespace sycl {

namespace detail {

template <int dimensions>
struct range_ {
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
};

} // namespace detail

template <int dimensions = 1>
class range;

template <>
struct range<1> : protected detail::range_<1> {
	range(size_t size)
		: detail::range_<1>(size, 1, 1) {}
};
template <>
struct range<2> : protected detail::range_<2> {
	range(size_t sizeX, size_t sizeY)
		: detail::range_<2>(sizeX, sizeY, 1) {}
	range(size_t size[2])
		: range(size[0], size[1]) {}
};
template <>
struct range<3> : protected detail::range_<3> {
	range(size_t sizeX, size_t sizeY, size_t sizeZ)
		: detail::range_<3>(sizeX, sizeY, sizeZ) {}
	range(size_t size[3])
		: range(size[0], size[1], size[2]) {}
};

template <int dimensions = 1>
class id {
public:
	id(range<dimensions> global_size, range<dimensions> local_size);
	int get(int dimension);
};

template <int dimensions>
class nd_range {
private:
	range<dimensions> global_size;
	range<dimensions> local_size;

public:
	nd_range(range<dimensions> global_size, range<dimensions> local_size)
		: global_size(global_size), local_size(local_size) {}

	nd_range(range<dimensions> global_size, range<dimensions> local_size, id offset);

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
