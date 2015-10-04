#pragma once

#include "../../data_ref.h"
#include "../../common.h"

namespace cl {
namespace sycl {
namespace detail {

#define SYCL_POINT_OP_EQ(lhs, op)				\
	for(size_t i = 0; i < dimensions; ++i) {	\
		lhs values[i] op= rhs.values[i];		\
	}

template <size_t dimensions>
struct point_base {
protected:
	friend class point_ref;

	data_ref::type_t type = data_ref::type_t::general;
	size_t values[dimensions];

	void set(point_base& rhs) {
		type = rhs.type;
		SYCL_POINT_OP_EQ(this->, );
	}

	void set(size_t value) {
		for(size_t i = 0; i < dimensions; ++i) {
			values[i] = value;
		}
	}
};

template <size_t dimensions, bool is_numeric = true>
struct point;

// Numeric points
template <size_t dimensions>
struct point<dimensions, true>
	: public point_base<dimensions>
{
	template <size_t dimensions, bool is_numeric>
	friend struct point;

	point& operator+=(const point& rhs) {
		SYCL_POINT_OP_EQ(this->, +);
		return *this;
	}
	template <bool is_numeric>
	point<dimensions, is_numeric> operator+(const point<dimensions, is_numeric>& rhs) const {
		point<dimensions, is_numeric> lhs;
		lhs.set(*this);
		SYCL_POINT_OP_EQ(lhs., +);
		return lhs;
	}

	template <bool defer = true>
	point<1, true> get(int dimension) const {
		point<1, true> value;
		value.type = type;
		value.values[0] = values[dimension];
		return value;
	}
	size_t& operator[](int dimension) {
		return values[dimension];
	}

	template <class one_dim = std::enable_if<dimensions == 1>::type>
	operator size_t() {
		return values[0];
	}
};

// Non-numeric points
template <size_t dimensions>
struct point<dimensions, false>
	: public point_base<dimensions>
{
	point& operator+=(const point<dimensions, true>& rhs) {
		SYCL_POINT_OP_EQ(this->, +);
		return *this;
	}
	point operator+(const point<dimensions, true>& rhs) const {
		point lhs;
		lhs.set(*this);
		SYCL_POINT_OP_EQ(lhs., +);
		return lhs;
	}
	point_ref operator+(const point& rhs) const {
		return point_ref(this) + rhs;
	}
};

#undef SYCL_POINT_OP_EQ

struct point_names {
	static const string_class all_suffix;

	static const string_class id_global;
	static const string_class range_global;
	
	static const string_class id_local;
	static const string_class range_local;
};

} // namespace detail
} // namespace sycl
} // namespace cl
