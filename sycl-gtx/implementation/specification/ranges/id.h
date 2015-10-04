#pragma once

// 3.5.1.3 ID class

#include "item.h"
#include "range.h"
#include "point.h"
#include "../../common.h"
#include "../../data_ref.h"
#include <initializer_list>

namespace cl {
namespace sycl {

// Forward declaration
template <int dims>
struct nd_item;

namespace detail {

// Forward declarations
namespace kernel_ {
	class source;
	template <class Input>
	struct constructor;
}

template <int dimensions>
struct id_ {
protected:
	friend class data_ref;
	friend struct kernel_::constructor<nd_item<dimensions>>;
	friend class kernel_::source;

	data_ref::type_t type;
	size_t values[3];

	id_(size_t first, size_t second, size_t third);

public:
	data_ref operator+(size_t n) const;
	data_ref operator-(size_t n) const;
	data_ref operator*(size_t n) const;
	data_ref operator/(size_t n) const;
	data_ref operator%(size_t n) const;

	data_ref operator>(size_t n) const;
	data_ref operator<(size_t n) const;
	data_ref operator>=(size_t n) const;
	data_ref operator<=(size_t n) const;
	data_ref operator==(size_t n) const;
	data_ref operator!=(size_t n) const;

	// TODO: More operators

	friend data_ref operator*(size_t n, id_<dimensions> i) {
		return i * n;
	}

	// Return the value of the specified dimension of the id
	size_t get(int n) const {
		return values[n];
	}
	id_ref operator[](size_t n);
};

} // namespace detail


template <int dimensions = 1>
struct id;

template <>
struct id<1> : detail::point<1, true> {
	template <class Input>
	friend struct detail::kernel_::constructor;
	friend class detail::data_ref;

	id(size_t x = 0) {
		values[0] = x;
	}
	id(const range<1>& rangeSize)
		: id(rangeSize.get(0)) {}
	id(const item<1>& rhs)
		: id(rhs.get()) {}
};

template <>
struct id<2> : detail::point<2, true> {
	template <class Input>
	friend struct detail::kernel_::constructor;
	friend class detail::data_ref;

	id(size_t x = 0, size_t y = 0) {
		values[0] = x;
		values[1] = y;
	}
	id(const range<2>& rangeSize)
		: id(rangeSize.get(0), rangeSize.get(1)) {}
	id(const item<2>& rhs)
		: id(rhs.get()) {}
};

template <>
struct id<3> : detail::point<3, true> {
	template <class Input>
	friend struct detail::kernel_::constructor;
	friend class detail::data_ref;

	id(size_t x = 0, size_t y = 0, size_t z = 0) {
		values[0] = x;
		values[1] = y;
		values[2] = z;
	}
	id(const range<3>& rangeSize)
		: id(rangeSize.get(0), rangeSize.get(1), rangeSize.get(2)) {}
	id(const item<3>& rhs)
		: id(rhs.get()) {}
};

} // namespace sycl
} // namespace cl
