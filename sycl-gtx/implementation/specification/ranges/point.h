#pragma once

#include "../../data_ref.h"
#include "../../common.h"

namespace cl {
namespace sycl {
namespace detail {

struct point_names {
	static const string_class all_suffix;

	static const string_class id_global;
	static const string_class range_global;

	static const string_class id_local;
	static const string_class range_local;
};


#define SYCL_POINT_OP_EQ(lhs, op)				\
	for(size_t i = 0; i < dimensions; ++i) {	\
		lhs values[i] op= rhs.values[i];		\
	}


template <size_t dimensions>
struct point : data_ref {
protected:
	friend class point_ref;
	template <size_t dimensions>
	friend struct point;

	size_t values[dimensions];

	void set(type_t type_) {
		type = type_;

		switch(type) {
			case type_t::id_global:
				name = point_names::id_global;
				break;
			case type_t::id_local:
				name = point_names::id_local;
				break;
			case type_t::range_global:
				name = point_names::range_global;
				break;
			case type_t::range_local:
				name = point_names::range_local;
				break;
		}
	}

	void set(point& rhs) {
		set(rhs.type);
		SYCL_POINT_OP_EQ(this->, );
	}

	void set(size_t value) {
		for(size_t i = 0; i < dimensions; ++i) {
			values[i] = value;
		}
	}

	point(size_t x, size_t y = 0, size_t z = 0)
		: data_ref("") {
		type = type_t::numeric;
		values[0] = x;
		if(dimensions > 1) {
			values[1] = y;
			if(dimensions > 2) {
				values[2] = z;
			}
		}
	}

public:
	point& operator+=(const point& rhs) {
		point lhs;
		lhs.set(*this);
		SYCL_POINT_OP_EQ(lhs., +);
	}
	point operator+(const point& rhs) const {
		return point_ref(this) + rhs;
	}

	point<1> get(int dimension) const {
		auto value = values[dimension];
		point<1> lhs(value);
		lhs.set(type);
		switch(type) {
			case type_t::id_global:
			case type_t::id_local:
			case type_t::range_global:
			case type_t::range_local:
				lhs.name += std::to_string(dimension);
				lhs.type = type_t::general;
				break;
		}
		return lhs;
	}

	// TODO: Not correct for non-numeric
	size_t& operator[](int dimension) {
		return values[dimension];
	}

	template <class one_dim = std::enable_if<dimensions == 1>::type>
	operator size_t() {
		return values[0];
	}
};

#undef SYCL_POINT_OP_EQ


} // namespace detail
} // namespace sycl
} // namespace cl
