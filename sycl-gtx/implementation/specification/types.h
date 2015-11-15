#pragma once

// 3.7 Data Types

#include "../common.h"
#include "../counter.h"
#include "../data_ref.h"

namespace cl {
namespace sycl {

// 3.7.2 Vector types

namespace detail {

// Forward declaration
template <typename dataT, int numElements>
class vec_base;

template <typename dataT, int numElements>
struct vec_members {
	vec_members(string_class name = "") {}
};

#define SYCL_V(member)	member(name + "." #member)

template <typename dataT>
struct vec_members<dataT, 2> : vec_members<dataT, 1>{
	vec_base<dataT, 1> x, y, s0, s1, lo, hi;
	vec_base<dataT, 2> xx, xy, yx, yy;
	vec_members(string_class name)
		:	SYCL_V(x), SYCL_V(y), SYCL_V(s0), SYCL_V(s1), SYCL_V(lo), SYCL_V(hi),
			SYCL_V(xx), SYCL_V(xy), SYCL_V(yx), SYCL_V(yy) {}
};

#undef SYCL_V

template <typename dataT, int numElements>
class vec_base : protected detail::counter<vec_base<dataT, numElements>>, public detail::data_ref {
private:
	template <typename dataT, int numElements>
	friend struct detail::vec_members;

	static string_class type_name() {
		return detail::type_string<dataT>() + (numElements == 1 ? "" : std::to_string(numElements));
	}

	string_class generate_name() const {
		return '_' + type_name() + '_' + std::to_string(get_count_id());
	}

	vec_base(string_class name)
		: data_ref(name) {}

public:
	vec_base()
		: counter(), data_ref(generate_name()) {
		detail::kernel_add(type_name() + ' ' + name);
	}

	template <class T>
	vec_base(T n)
		: counter(), data_ref(generate_name()) {
		detail::kernel_add(type_name() + ' ' + name + " = " + get_name(n));
	}

	// TODO: Swizzle methods
	//swizzled_vec<T, out_dims> swizzle<int s1, ...>();
#ifdef SYCL_SIMPLE_SWIZZLES
	swizzled_vec<T, 4> xyzw();
	...
#endif // #ifdef SYCL_SIMPLE_SWIZZLES
};

} // namespace detail


#define SYCL_VEC_INHERIT_CONSTRUCTORS(type)	\
	type()									\
		: Base() {}							\
	template <class T>						\
	type(T n)								\
		: Base(n) {}						\

#define SYCL_VEC_INHERIT_ASSIGNMENTS(type)			\
	template <class T>								\
	data_ref& operator=(T n) {						\
		return data_ref::operator=(n);				\
	}												\
	template <class T>								\
	friend data_ref operator*(T n, type&& elem) {	\
		return n * (data_ref&&)elem;				\
	}

template <typename dataT, int numElements>
class vec : public detail::vec_base<dataT, numElements>, public detail::vec_members<dataT, numElements> {
private:
	using Base = detail::vec_base<dataT, numElements>;
	using Members = detail::vec_members<dataT, numElements>;
public:
	vec()
		: Base(), Members(name) {}
	template <class T>
	vec(T n)
		: Base(n), Members(name) {}
	SYCL_VEC_INHERIT_ASSIGNMENTS(vec)
};


// 3.9.1 Description of the built-in types available for SYCL host and device

#define SYCL_VEC_SIGNED(type, numElements)						\
	class type##numElements : public vec<type, numElements> {	\
		using Base = vec<type, numElements>;					\
	public:														\
		SYCL_VEC_INHERIT_CONSTRUCTORS(type##numElements)		\
		SYCL_VEC_INHERIT_ASSIGNMENTS(type##numElements)			\
	};

#define SYCL_VEC_UNSIGNED(type, numElements)								\
	class u##type##numElements : public vec<unsigned type, numElements> {	\
		using Base = vec<unsigned type, numElements>;						\
	public:																	\
		SYCL_VEC_INHERIT_CONSTRUCTORS(u##type##numElements)					\
		SYCL_VEC_INHERIT_ASSIGNMENTS(u##type##numElements)					\
	};

#define SYCL_ADD_SIGNED_vec(type)	\
	SYCL_VEC_SIGNED(type, 1)		\
	SYCL_VEC_SIGNED(type, 2)		\
	SYCL_VEC_SIGNED(type, 3)		\
	SYCL_VEC_SIGNED(type, 4)		\
	SYCL_VEC_SIGNED(type, 8)		\
	SYCL_VEC_SIGNED(type, 16)

#define SYCL_ADD_UNSIGNED_vec(type)	\
	SYCL_VEC_UNSIGNED(type, 1)		\
	SYCL_VEC_UNSIGNED(type, 2)		\
	SYCL_VEC_UNSIGNED(type, 3)		\
	SYCL_VEC_UNSIGNED(type, 4)		\
	SYCL_VEC_UNSIGNED(type, 8)		\
	SYCL_VEC_UNSIGNED(type, 16)

SYCL_ADD_SIGNED_vec(bool)
SYCL_ADD_SIGNED_vec(int)
SYCL_ADD_SIGNED_vec(char)
SYCL_ADD_SIGNED_vec(short)
SYCL_ADD_SIGNED_vec(long)
SYCL_ADD_SIGNED_vec(float)
SYCL_ADD_SIGNED_vec(double)

SYCL_ADD_UNSIGNED_vec(int)
SYCL_ADD_UNSIGNED_vec(char)
SYCL_ADD_UNSIGNED_vec(short)
SYCL_ADD_UNSIGNED_vec(long)

#undef SYCL_ADD_SIGNED_vec
#undef SYCL_ADD_UNSIGNED_vec
#undef SYCL_VEC_INHERIT_ASSIGNMENTS
#undef SYCL_VEC_INHERIT_CONSTRUCTORS
#undef SYCL_VEC_SIGNED
#undef SYCL_VEC_UNSIGNED

} // namespace sycl
} // namespace cl
