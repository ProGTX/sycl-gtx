#pragma once

#include "cl_type.h"
#include "../../common.h"
#include <initializer_list>


namespace cl {
namespace sycl {

namespace detail {
namespace vectors {

// Forward declaration
template <typename, int>
class base;

template <typename dataT, int parentElems, int selfElems = parentElems>
struct cl_base;

template <typename dataT>
struct cl_base<dataT, 1, 1> {
private:
	dataT elem;
public:
	operator dataT&() {
		return elem;
	}
	operator const dataT&() const {
		return elem;
	}
};

template <typename dataT, int parentElems>
struct cl_base<dataT, parentElems, 0> {
protected:
	template <typename, int, int>
	friend struct cl_base;
	template <typename>
	friend struct type_string;
	template <typename, int>
	friend class base;

	using genvector = typename detail::cl_type<dataT, parentElems>::type;

	dataT elems[parentElems];

	static string_class type_name() {
		return type_string<dataT>::get() + (parentElems == 1 ? "" : get_string<int>::get(parentElems));
	}

public:
	operator genvector&() {
		return *reinterpret_cast<genvector*>(elems);
	}
	operator const genvector&() const {
		return *reinterpret_cast<genvector*>(elems);
	}
};

template <typename dataT, int parentElems>
struct cl_base<dataT, parentElems, 2> : cl_base<dataT, parentElems, 0> {
	dataT& x() {
		return this->elems[0];
	}
	const dataT& x() const {
		return this->elems[0];
	}
	dataT& y() {
		return this->elems[1];
	}
	const dataT& y() const {
		return this->elems[1];
	}
};

template <typename dataT, int parentElems>
struct cl_base<dataT, parentElems, 3> : cl_base<dataT, parentElems, 2> {
	dataT& z() {
		return this->elems[2];
	}
	const dataT& z() const {
		return this->elems[2];
	}

	cl_base<dataT, parentElems, 3> xyz() {
		return *this;
	}
};

template <typename dataT, int parentElems>
struct cl_base<dataT, parentElems, 4> : cl_base<dataT, parentElems, 3> {
protected:
	using cl_base_3 = cl_base<dataT, 3, 3>;

public:
	dataT& w() {
		return this->elems[3];
	}
	const dataT& w() const {
		return this->elems[3];
	}

	operator cl_base_3&() {
		return *reinterpret_cast<cl_base_3*>(this);
	}
	operator const cl_base_3&() const {
		return *reinterpret_cast<cl_base_3*>(this);
	}
};

template <typename dataT, int parentElems>
struct cl_base<dataT, parentElems, 8> : cl_base<dataT, parentElems, 0> {
	cl_base<dataT, 4, 4> lo() {
		return *reinterpret_cast<cl_base<dataT, 4, 4>*>(this);
	}
	cl_base<dataT, 4, 4> hi() {
		cl_base<dataT, 4, 4> ret;
		using type = decltype(elems + 0);
		reinterpret_cast<type&>(ret.elems) = reinterpret_cast<type>(elems + 4);
		return ret;
	}
};

template <typename dataT, int parentElems>
struct cl_base<dataT, parentElems, 16> : cl_base<dataT, parentElems, 0> {
	cl_base<dataT, 8, 8> lo() {
		return *reinterpret_cast<cl_base<dataT, 8, 8>*>(this);
	}
	cl_base<dataT, 8, 8> hi() {
		cl_base<dataT, 8, 8> ret;
		using type = decltype(elems + 0);
		reinterpret_cast<type&>(ret.elems) = reinterpret_cast<type>(elems + 8);
		return ret;
	}
};

} // namespace vectors
} // namespace detail

} // namespace sycl
} // namespace cl
