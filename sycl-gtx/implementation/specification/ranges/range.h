#pragma once

// 3.5.1.1 Range class

#include "point.h"

namespace cl {
namespace sycl {

template <int dimensions = 1>
struct range;

template <>
struct range<1> : detail::point<1, true> {
	range(size_t x) {
		values[0] = x;
	}
	size_t size() const {
		return values[0];
	}
};

template <>
struct range<2> : detail::point<2, true> {
	range(size_t x, size_t y) {
		values[0] = x;
		values[1] = y;
	}
	size_t size() const {
		return values[0] * values[1];
	}
};

template <>
struct range<3> : detail::point<3, true> {
	range(size_t x, size_t y, size_t z) {
		values[0] = x;
		values[1] = y;
		values[2] = z;
	}
	size_t size() const {
		return values[0] * values[1] * values[2];
	}
};


namespace detail {

template <int dimensions>
static range<dimensions> empty_range();
template <>
static range<1> empty_range() {
	return range<1>(0);
}
template <>
static range<2> empty_range() {
	return range<2>(0, 0);
}
template <>
static range<3> empty_range() {
	return range<3>(0, 0, 0);
}

} // namespace detail

} // namespace sycl
} // namespace cl
