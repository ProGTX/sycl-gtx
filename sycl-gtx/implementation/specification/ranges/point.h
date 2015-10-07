#pragma once

#include "point_ref.h"
#include "../../data_ref.h"
#include "../../common.h"

namespace cl {
namespace sycl {
namespace detail {

struct point_names {
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
	friend class data_ref;
	template <size_t dimensions>
	friend struct point;

	template <int dimensions>
	friend struct get_special_id;
	template <int dimensions>
	friend struct get_special_range;
	template <int dimensions, bool is_id>
	friend struct identifier_code;

	static string_class name_from_type(type_t type) {
		string_class name = "";
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
		return name;
	}

	size_t values[dimensions];

	void set(type_t type_) {
		type = type_;
		name = name_from_type(type);
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

	bool is_identifier() const {
		switch(type) {
			case type_t::id_global:
			case type_t::id_local:
			case type_t::range_global:
			case type_t::range_local:
				return true;
			default:
				return false;
		}
	}

	point(size_t x, size_t y, size_t z)
		: data_ref("") {
		type = type_t::numeric;
		values[0] = x;
		if(dimensions > 1) {
			values[1] = y;
			if(dimensions > 2) {
				values[2] = z;
			}
		}
		else {
			name = std::to_string(x);
		}
	}

public:
	point& operator+=(const data_ref& rhs) {
		set(type_t::general);
		return data_ref::operator+=(rhs);
	}
	point& operator+=(const point& rhs) {
		if(type == type_t::numeric && rhs.type == type_t::numeric) {
			SYCL_POINT_OP_EQ(this->, +);
			if(dimensions == 1) {
				name = std::to_string(values[0]);
			}
		}
		else {
			return operator+=((data_ref)rhs);
		}
	}

	point operator+(const point& rhs) const {
		point lhs(*this);
		if(type == type_t::numeric && rhs.type == type_t::numeric) {
			SYCL_POINT_OP_EQ(lhs., +);
		}
		else {
			lhs.set(type_t::general);
			((data_ref)lhs).operator+=(rhs);
		}
		return lhs;
	}
	data_ref operator+(const data_ref& rhs) const {
		return data_ref::operator+(rhs);
	}

private:
	template <bool is_const>
	point_ref<is_const> get_ref(int dimension) {
		auto name_ = name;
		bool is_ident = is_identifier();
		if(is_ident) {
			name_ += std::to_string(dimension);
		}
		return point_ref<is_const>(values[dimension], name_, type);
	}

public:
	point_ref<true> get(int dimension) const {
		// The const cast is ugly, but the get_ref method doesn't actually change the this pointer
		return const_cast<point<dimensions>*>(this)->get_ref<true>(dimension);
	}
	point_ref<false> operator[](int dimension) {
		return get_ref<false>(dimension);
	}
};

#undef SYCL_POINT_OP_EQ

} // namespace detail
} // namespace sycl
} // namespace cl
