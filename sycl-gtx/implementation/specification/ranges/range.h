#pragma once

// 3.7.1.1 Range class

#include "../../common.h"
#include <initializer_list>

namespace cl {
namespace sycl {

namespace detail {

template <int dimensions>
struct range_ {
protected:
	size_t dims[3];

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
	size_t& operator[](size_t n) {
		return dims[n];
	}
	size_t operator[](size_t n) const {
		return dims[n];
	}

	// Return the value of the specified dimension of the range.
	// TODO: Operator[] should take care of this
	size_t get(int dimension) const {
		return dims[n];
	}
};

} // namespace detail

template <int dimensions = 1>
struct range;

#define sycl_range_op(op, dimensions)				\
	range operator op(range b) {					\
		range c((const range&)(*this));				\
		for(::size_t i = 0; i < dimensions; ++i) {	\
			c.dims[i] op= b.dims[i];				\
		}											\
		return c;									\
	}

#define sycl_range_operators(dimensions)	\
	sycl_range_op(*, dimensions)			\
	sycl_range_op(/, dimensions)			\
	sycl_range_op(+, dimensions)			\
	sycl_range_op(-, dimensions)

template <>
struct range<1> : detail::range_<1> {
	range(size_t size)
		: detail::range_<1>(size, 1, 1) {}
	range(size_t size[1])
		: range(size[0]) {}
	sycl_range_operators(1);
};
template <>
struct range<2> : detail::range_<2>{
	range(size_t sizeX, size_t sizeY)
		: detail::range_<2>(sizeX, sizeY, 1) {}
	range(size_t size[2])
		: range(size[0], size[1]) {}
	sycl_range_operators(2);
};
template <>
struct range<3> : detail::range_<3>{
	range(size_t sizeX, size_t sizeY, size_t sizeZ)
		: detail::range_<3>(sizeX, sizeY, sizeZ) {}
	range(size_t size[3])
		: range(size[0], size[1], size[2]) {}
	sycl_range_operators(3);
};

#define sycl_static_range_ops(op)						\
template <int dims>										\
range<dims> operator op(range<dims> a, range<dims> b) {	\
	return a op b;										\
}

sycl_static_range_ops(*);
sycl_static_range_ops(/);
sycl_static_range_ops(+);
sycl_static_range_ops(-);

namespace detail {

template<int dimensions>
static range<dimensions> empty_range() {
	return range<dimensions>(vector_class<size_t>(dimensions, 0).data());
};

}


} // namespace sycl
} // namespace cl

#undef sycl_range_op
#undef sycl_range_operators
#undef sycl_static_range_ops
