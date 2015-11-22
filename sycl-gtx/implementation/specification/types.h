#pragma once

// 3.7 Data Types

#include "vector_members.h"

#include "../common.h"
#include "../counter.h"
#include "../data_ref.h"

namespace cl {
namespace sycl {

// 3.7.2 Vector types

// Forward declaration
template <typename dataT, int numElements>
class vec;

namespace detail {
namespace vectors {

// Forward declaration
template <typename dataT, int numElements>
struct data;

#define SYCL_ENABLE_IF_DIM(dim)	\
typename std::enable_if<num == dim>::type* = nullptr


template <typename dataT, int numElements>
class base : protected counter<base<dataT, numElements>>, public data_ref {
private:
	template <typename, int>
	friend struct members;
	template <typename>
	friend struct type_string;

	static string_class type_name() {
		return type_string<dataT>::get() + (numElements == 1 ? "" : std::to_string(numElements));
	}

	string_class generate_name() const {
		return '_' + type_name() + '_' + std::to_string(get_count_id());
	}

	string_class this_name() const {
		return type_name() + ' ' + name;
	}

protected:
	base(string_class assign, bool generate_new)
		: data_ref(generate_name()) {
		kernel_add(this_name() + " = " + assign);
	}

	base(string_class name)
		: data_ref(name) {}

public:
	base()
		: data_ref(generate_name()) {
		kernel_add(this_name());
	}

	template <class T>
	base(T n, typename std::enable_if<!std::is_same<T, const base&>::value>::type* = nullptr)
		: base(get_name(n), true) {}

	template <int num = numElements>
	base(data_ref x, data_ref y, SYCL_ENABLE_IF_DIM(2))
		: base(open_parenthesis + x.name + ", " + y.name + ')', true) {}

	template <int num = numElements>
	base(data_ref x, data_ref y, data_ref z, SYCL_ENABLE_IF_DIM(3))
		: base(open_parenthesis + x.name + ", " + y.name + ", " + z.name + ')', true) {}

	operator vec<dataT, numElements>&() {
		return *reinterpret_cast<vec<dataT, numElements>*>(this);
	}

	size_t get_count() const {
		return numElements;
	}
	size_t get_size() const {
		return numElements * sizeof(typename data<dataT, numElements>::type);
	}

	// TODO: Swizzle methods
	//swizzled_vec<T, out_dims> swizzle<int s1, ...>();
#ifdef SYCL_SIMPLE_SWIZZLES
	swizzled_vec<T, 4> xyzw();
	...
#endif // #ifdef SYCL_SIMPLE_SWIZZLES
};

} // namespace vectors


template <typename dataT, int numElements>
struct data_size<vec<dataT, numElements>> {
	static size_t get() {
		return sizeof(typename base_host_data<vec<dataT, numElements>>::type);
	}
};

template <typename dataT, int numElements>
struct base_host_data<vec<dataT, numElements>> {
	using type = typename vectors::data<dataT, numElements>::type;
};

} // namespace detail


template <typename dataT, int numElements>
class vec : public detail::vectors::base<dataT, numElements>, public detail::vectors::members<dataT, numElements> {
private:
	template <typename, int>
	friend struct detail::vectors::members;

	using Base = detail::vectors::base<dataT, numElements>;
	using Members = detail::vectors::members<dataT, numElements>;

protected:
	vec(string_class name_)
		: Base(name_), Members(this, name) {}
public:
	vec()
		: Base(), Members(this, name) {}
	vec(const vec& copy)
		: Base(copy.name, true), Members(this, name) {}
	template <class T>
	vec(T n, typename std::enable_if<!std::is_same<T, const vec&>::value>::type* = nullptr)
		: Base(n), Members(this, name) {}
	template <int num = numElements>
	vec(data_ref x, data_ref y, SYCL_ENABLE_IF_DIM(2))
		: Base(x, y), Members(this, name) {}
	template <int num = numElements>
	vec(data_ref x, data_ref y, data_ref z, SYCL_ENABLE_IF_DIM(3))
		: Base(x, y, z), Members(this, name) {}

	template <class T>
	vec& operator=(T&& n) {
		data_ref::operator=(std::forward<T>(n));
		return *this;
	}
};


// 3.10.1 Description of the built-in types available for SYCL host and device

#define SYCL_VEC_SIGNED(basetype, numElements) \
using basetype##numElements = vec<basetype, numElements>;

#define SYCL_VEC_SIGNED_EXTRA(basetype, numElements)	\
SYCL_VEC_SIGNED(basetype, numElements)					\
template <>												\
struct detail::vectors::data<basetype, numElements> {	\
	using type = cl_##basetype##numElements;			\
};

#define SYCL_VEC_SIGNED_EXTRA_ONE(basetype)	\
SYCL_VEC_SIGNED(basetype, 1)				\
template <>									\
struct detail::vectors::data<basetype, 1> {	\
	using type = cl_##basetype;				\
};

#define SYCL_VEC_UNSIGNED(type, numElements) \
using u##type##numElements = vec<unsigned type, numElements>;

#define SYCL_ADD_SIGNED_vec(type)	\
	SYCL_VEC_SIGNED_EXTRA(type, 2)	\
	SYCL_VEC_SIGNED_EXTRA(type, 3)	\
	SYCL_VEC_SIGNED_EXTRA(type, 4)	\
	SYCL_VEC_SIGNED_EXTRA(type, 8)	\
	SYCL_VEC_SIGNED_EXTRA(type, 16)

#define SYCL_ADD_UNSIGNED_vec(type)	\
	SYCL_VEC_UNSIGNED(type, 1)		\
	SYCL_VEC_UNSIGNED(type, 2)		\
	SYCL_VEC_UNSIGNED(type, 3)		\
	SYCL_VEC_UNSIGNED(type, 4)		\
	SYCL_VEC_UNSIGNED(type, 8)		\
	SYCL_VEC_UNSIGNED(type, 16)

SYCL_ADD_SIGNED_vec(int)
SYCL_ADD_SIGNED_vec(char)
SYCL_ADD_SIGNED_vec(short)
SYCL_ADD_SIGNED_vec(long)
SYCL_ADD_SIGNED_vec(float)
SYCL_ADD_SIGNED_vec(double)

SYCL_VEC_SIGNED_EXTRA_ONE(int)
SYCL_VEC_SIGNED_EXTRA_ONE(char)
SYCL_VEC_SIGNED_EXTRA_ONE(short)
SYCL_VEC_SIGNED_EXTRA_ONE(long)
SYCL_VEC_SIGNED_EXTRA_ONE(float)
SYCL_VEC_SIGNED_EXTRA_ONE(double)

SYCL_VEC_SIGNED(bool, 1)
template <>
struct detail::vectors::data<bool, 1> {
	using type = cl_bool;
};

SYCL_ADD_UNSIGNED_vec(int)
SYCL_ADD_UNSIGNED_vec(char)
SYCL_ADD_UNSIGNED_vec(short)
SYCL_ADD_UNSIGNED_vec(long)

#undef SYCL_ENABLE_IF_DIM
#undef SYCL_ADD_SIGNED_vec
#undef SYCL_ADD_UNSIGNED_vec
#undef SYCL_VEC_SIGNED
#undef SYCL_VEC_SIGNED_EXTRA
#undef SYCL_VEC_SIGNED_EXTRA_ONE
#undef SYCL_VEC_UNSIGNED

} // namespace sycl
} // namespace cl
