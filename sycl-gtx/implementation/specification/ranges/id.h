#pragma once

// TODO: 3.7.1.3 ID class

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
	id_ref operator[](size_t n);

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

	friend data_ref operator*(size_t n, id_<dimensions> i) {
		return i * n;
	}

	// Return the value of the specified dimension of the id
	size_t get(int n) const {
		return values[n];
	}
};

} // namespace detail

template <int dimensions = 1>
struct id;

template <>
struct id<1> : detail::id_<1> {
	id(size_t size = 0)
		: detail::id_<1>(size, 1, 1) {}
	id(std::initializer_list<size_t> list)
		: id(*(list.begin())) {}
	id(size_t size[1])
		: id(size[0]) {}
};
template <>
struct id<2> : detail::id_<2>{
	id(size_t sizeX, size_t sizeY)
		: detail::id_<2>(sizeX, sizeY, 1) {}
	id()
		: id(0, 0) {}
	id(std::initializer_list<size_t> list)
		: id(*(list.begin()), *(list.begin() + 1)) {}
	id(size_t size[2])
		: id(size[0], size[1]) {}
};
template <>
struct id<3> : detail::id_<3>{
	id(size_t sizeX, size_t sizeY, size_t sizeZ)
		: detail::id_<3>(sizeX, sizeY, sizeZ) {}
	id()
		: id(0, 0, 0) {}
	id(std::initializer_list<size_t> list)
		: id(*(list.begin()), *(list.begin() + 1), *(list.begin() + 2)) {}
	id(size_t size[3])
		: id(size[0], size[1], size[2]) {}
};

} // namespace sycl
} // namespace cl
