#pragma once

// 3.5.1.3 ID class

#include "range.h"
#include "item.h"
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
	template<class Input>
	struct constructor;
}

static const string_class id_global_name		= "_sycl_id_";
static const string_class id_global_all_name	= "_sycl_id_d";
static const string_class id_local_name			= "_sycl_id_local_";
static const string_class id_local_all_name		= "_sycl_id_local_d";


template <int dimensions>
struct id_ {
protected:
	friend class data_ref;
	friend struct kernel_::constructor<nd_item<dimensions>>;
	friend class kernel_::source;

	id_ref::type type;
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

// TODO
template <int dimensions = 1>
using index = id<dimensions>;

template <>
struct id<1> : detail::id_<1> {
	id(size_t size = 0)
		: detail::id_<1>(size, 1, 1) {}
	id(std::initializer_list<size_t> list)
		: id(*(list.begin())) {}
	id(const range<1>& rangeSize)
		: id(rangeSize.get(0)) {}
	id(const item<1>& it)
		: id(it.get(0)) {}
	operator size_t() {
		return values[0];
	}
};
template <>
struct id<2> : detail::id_<2>{
	id(size_t sizeX, size_t sizeY)
		: detail::id_<2>(sizeX, sizeY, 1) {}
	id()
		: id(0, 0) {}
	id(std::initializer_list<size_t> list)
		: id(*(list.begin()), *(list.begin() + 1)) {}
	id(const range<2>& rangeSize)
		: id(rangeSize.get(0), rangeSize.get(1)) {}
	id(const item<2>& it)
		: id(it.get(0), it.get(1)) {}
};
template <>
struct id<3> : detail::id_<3>{
	id(size_t sizeX, size_t sizeY, size_t sizeZ)
		: detail::id_<3>(sizeX, sizeY, sizeZ) {}
	id()
		: id(0, 0, 0) {}
	id(std::initializer_list<size_t> list)
		: id(*(list.begin()), *(list.begin() + 1), *(list.begin() + 2)) {}
	id(const range<3>& rangeSize)
		: id(rangeSize.get(0), rangeSize.get(1), rangeSize.get(2)) {}
	id(const item<3>& it)
		: id(it.get(0), it.get(1), it.get(2)) {}
};

} // namespace sycl
} // namespace cl
