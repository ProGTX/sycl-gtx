#pragma once

// 3.7 Data Types

#include "access.h"
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
	friend struct ::cl::sycl::detail::type_string;

	static string_class type_name() {
		return type_string<dataT>::get() + (numElements == 1 ? "" : get_string<int>::get(numElements));
	}

	string_class generate_name() const {
		return '_' + type_name() + '_' + get_string<counter_t>::get(this->get_count_id());
	}

	string_class this_name() const {
		return type_name() + ' ' + this->name;
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

	base(const base&) = default;
	base& operator=(const base&) = default;

	template <class T>
	base(T n, typename std::enable_if<!std::is_same<T, const base&>::value>::type* = nullptr)
		: base(get_name(n), true) {}

	template <int num = numElements>
	base(data_ref x, data_ref y, SYCL_ENABLE_IF_DIM(2))
		: base(open_parenthesis + type_name() + ")(" + x.name + ", " + y.name + ')', true) {}

	template <int num = numElements>
	base(data_ref x, data_ref y, data_ref z, SYCL_ENABLE_IF_DIM(3))
		: base(open_parenthesis + type_name() + ")(" + x.name + ", " + y.name + ", " + z.name + ')', true) {}

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
} // namespace detail


template <typename dataT, int numElements>
class vec : public detail::vectors::base<dataT, numElements>, public detail::vectors::members<dataT, numElements> {
private:
	template <typename, int>
	friend struct detail::vectors::members;
	template <typename, int, detail::acc_mode_t, detail::acc_target_t, typename>
	friend class detail::accessor_;
	template <int, typename, int, access::mode, access::target>
	friend class detail::accessor_device_ref;

	using Base = detail::vectors::base<dataT, numElements>;
	using Members = detail::vectors::members<dataT, numElements>;

protected:
	vec(string_class name_)
		: Base(name_), Members(this) {}

	using data_ref = detail::data_ref;
public:
	vec()
		: Base(), Members(this) {}
	vec(const vec& copy)
		: Base(copy.name, true), Members(this) {}
	// TODO: Move members
	vec(vec&& move)
		: Base(std::move(move.name)), Members(this) {}
	template <class T>
	vec(T n, typename std::enable_if<!std::is_same<T, const vec&>::value>::type* = nullptr)
		: Base(n), Members(this) {}
	template <int num = numElements>
	vec(data_ref x, data_ref y, SYCL_ENABLE_IF_DIM(2))
		: Base(x, y), Members(this) {}
	template <int num = numElements>
	vec(data_ref x, data_ref y, data_ref z, SYCL_ENABLE_IF_DIM(3))
		: Base(x, y, z), Members(this) {}
	vec& operator=(const vec&) = default;
	template <class T>
	vec& operator=(T&& n) {
		data_ref::operator=(std::forward<T>(n));
		return *this;
	}
};


// 3.10.1 Description of the built-in types available for SYCL host and device

#define SYCL_VEC_SIGNED(basetype, numElements) \
using basetype##numElements = vec<basetype, numElements>;
#define SYCL_VEC_UNSIGNED(type, numElements) \
using u##type##numElements = vec<unsigned type, numElements>;

#define SYCL_VEC_SIGNED_EXTRA(basetype, numElements)		\
	SYCL_VEC_SIGNED(basetype, numElements)					\
	template <>												\
	struct detail::vectors::data<basetype, numElements> {	\
		using type = cl_##basetype##numElements;			\
	};
#define SYCL_ADD_SIGNED_SCALAR_vec(basetype)				\
	SYCL_VEC_SIGNED(basetype, 1)							\
	template <>												\
	struct detail::vectors::data<basetype, 1> {				\
		using type = cl_##basetype;							\
	};

#define SYCL_VEC_UNSIGNED_EXTRA(basetype, numElements)				\
	SYCL_VEC_UNSIGNED(basetype, numElements)						\
	template <>														\
	struct detail::vectors::data<unsigned basetype, numElements> {	\
		using type = cl_u##basetype##numElements;					\
	};
#define SYCL_ADD_UNSIGNED_SCALAR_vec(basetype)						\
	SYCL_VEC_UNSIGNED(basetype, 1)									\
	template <>														\
	struct detail::vectors::data<unsigned basetype, 1> {			\
		using type = cl_u##basetype;								\
	};

#define SYCL_ADD_SIGNED_vec(type)		\
	SYCL_VEC_SIGNED_EXTRA(type, 2)		\
	SYCL_VEC_SIGNED_EXTRA(type, 3)		\
	SYCL_VEC_SIGNED_EXTRA(type, 4)		\
	SYCL_VEC_SIGNED_EXTRA(type, 8)		\
	SYCL_VEC_SIGNED_EXTRA(type, 16)
#define SYCL_ADD_UNSIGNED_vec(type)		\
	SYCL_VEC_UNSIGNED_EXTRA(type, 2)	\
	SYCL_VEC_UNSIGNED_EXTRA(type, 3)	\
	SYCL_VEC_UNSIGNED_EXTRA(type, 4)	\
	SYCL_VEC_UNSIGNED_EXTRA(type, 8)	\
	SYCL_VEC_UNSIGNED_EXTRA(type, 16)

SYCL_ADD_SIGNED_vec(int)
SYCL_ADD_SIGNED_vec(char)
SYCL_ADD_SIGNED_vec(short)
SYCL_ADD_SIGNED_vec(long)
SYCL_ADD_SIGNED_vec(float)
SYCL_ADD_SIGNED_vec(double)

SYCL_ADD_SIGNED_SCALAR_vec(bool)
SYCL_ADD_SIGNED_SCALAR_vec(int)
SYCL_ADD_SIGNED_SCALAR_vec(char)
SYCL_ADD_SIGNED_SCALAR_vec(short)
SYCL_ADD_SIGNED_SCALAR_vec(long)
SYCL_ADD_SIGNED_SCALAR_vec(float)
SYCL_ADD_SIGNED_SCALAR_vec(double)

SYCL_ADD_UNSIGNED_vec(int)
SYCL_ADD_UNSIGNED_vec(char)
SYCL_ADD_UNSIGNED_vec(short)
SYCL_ADD_UNSIGNED_vec(long)

SYCL_ADD_UNSIGNED_SCALAR_vec(int)
SYCL_ADD_UNSIGNED_SCALAR_vec(char)
SYCL_ADD_UNSIGNED_SCALAR_vec(short)
SYCL_ADD_UNSIGNED_SCALAR_vec(long)

#undef SYCL_ENABLE_IF_DIM
#undef SYCL_VEC_SIGNED
#undef SYCL_VEC_SIGNED_EXTRA
#undef SYCL_ADD_SIGNED_vec
#undef SYCL_ADD_SIGNED_SCALAR_vec
#undef SYCL_VEC_UNSIGNED
#undef SYCL_VEC_UNSIGNED_EXTRA
#undef SYCL_ADD_UNSIGNED_vec
#undef SYCL_ADD_UNSIGNED_SCALAR_vec


namespace detail {

// These are defined elsewhere, here only specialization for vectors

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

template <typename dataT, int numElements>
struct acc_device_return<vec<dataT, numElements>> {
	using type = vec<dataT, numElements>;
};

// Cannot be joined with vector declarations
// because a vector of 3 is a typedef of a vector of 4
#define SYCL_CY_TYPE_STRING(nummedType)	\
template <>								\
struct type_string<cl_##nummedType> {	\
	static string_class get() {			\
		return #nummedType;				\
	}									\
};

#define SYCL_ADD_CL_TYPE_STRING(basetype)	\
	SYCL_CY_TYPE_STRING(basetype##2)		\
	SYCL_CY_TYPE_STRING(basetype##4)		\
	SYCL_CY_TYPE_STRING(basetype##8)		\
	SYCL_CY_TYPE_STRING(basetype##16)

SYCL_ADD_CL_TYPE_STRING(int)
SYCL_ADD_CL_TYPE_STRING(char)
SYCL_ADD_CL_TYPE_STRING(short)
SYCL_ADD_CL_TYPE_STRING(long)
SYCL_ADD_CL_TYPE_STRING(float)
SYCL_ADD_CL_TYPE_STRING(double)

SYCL_ADD_CL_TYPE_STRING(uint)
SYCL_ADD_CL_TYPE_STRING(uchar)
SYCL_ADD_CL_TYPE_STRING(ushort)
SYCL_ADD_CL_TYPE_STRING(ulong)

#undef SYCL_ADD_CL_TYPE_STRING
#undef SYCL_CY_TYPE_STRING

} // namespace detail

} // namespace sycl
} // namespace cl
