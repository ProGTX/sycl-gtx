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
	cl_base() {}
	cl_base(dataT value)
		: elem(value) {}

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
	cl_base() {}
	cl_base(const cl_base&) = default;
	cl_base(genvector v) {
		auto start = reinterpret_cast<dataT*>(&v);
		std::copy(start, start + parentElems, this->elems);
	}
#if MSVC_LOW							
	cl_base(cl_base&& move) {
		std::swap(this->elems, move.elems);
	}
	cl_base& operator=(cl_base&& move) {
		std::swap(this->elems, move.elems);
		return *this;
	}
#else
	cl_base(cl_base&&) = default;
	cl_base& operator=(cl_base&&) = default;
#endif
	cl_base& operator=(const cl_base&) = default;
	cl_base& operator=(genvector v) {
		std::copy(&v, &v + parentElems, this->elems);
	}

	operator genvector&() {
		return *reinterpret_cast<genvector*>(elems);
	}
	operator const genvector&() const {
		return *reinterpret_cast<genvector*>(elems);
	}
};


#define SYCL_CL_VEC_INHERIT_CONSTRUCTORS	\
	cl_base() {}							\
	cl_base(const cl_base&) = default;		\
	cl_base(genvector v)					\
		: Base(v) {}						\
	cl_base(cl_base&& move) {				\
		std::swap(this->elems, move.elems);	\
	}


template <typename dataT, int parentElems>
struct cl_base<dataT, parentElems, 2> : cl_base<dataT, parentElems, 0> {
protected:
	using Base = cl_base<dataT, parentElems, 0>;
public:
#if MSVC_LOW
	SYCL_CL_VEC_INHERIT_CONSTRUCTORS
#else
	using Base::Base;
#endif

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
protected:
	using Base = cl_base<dataT, parentElems, 2>;
public:
#if MSVC_LOW
	SYCL_CL_VEC_INHERIT_CONSTRUCTORS
#else
	using Base::Base;
#endif

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
private:
	using Base = cl_base<dataT, parentElems, 3>;
	using cl_base_3 = Base::cl_base_3;
public:
#if MSVC_LOW
	SYCL_CL_VEC_INHERIT_CONSTRUCTORS
#else
	using Base::Base;
#endif

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
private:
	using Base = cl_base<dataT, parentElems, 0>;
public:
#if MSVC_LOW
	SYCL_CL_VEC_INHERIT_CONSTRUCTORS
#else
	using Base::Base;
#endif

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
private:
	using Base = cl_base<dataT, parentElems, 0>;
public:
#if MSVC_LOW
	SYCL_CL_VEC_INHERIT_CONSTRUCTORS
#else
	using Base::Base;
#endif

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
