#pragma once

// 3.7 Data Types

#include "../common.h"
#include "../counter.h"
#include "../data_ref.h"

namespace cl {
namespace sycl {

// 3.7.2 Vector types

namespace detail {

#define SYCL_ENABLE_IF_DIM(dim)	\
typename std::enable_if<num == dim>::type* = nullptr

// Forward declaration
template <typename dataT, int numElements>
class vec_base;

template <typename dataT, int numElements>
struct vec_members {
	vec_members(string_class name = "") {}
};

#define SYCL_V(member)								member(name + "." #member)
#define SYCL_V2(m1, m2)								SYCL_V(m1), SYCL_V(m2)
#define SYCL_V3(m1, m2, m3)							SYCL_V2(m1, m2), SYCL_V(m3)
#define SYCL_V4(m1, m2, m3, m4)						SYCL_V2(m1, m2), SYCL_V2(m3, m4)
#define SYCL_V5(m1, m2, m3, m4, m5)					SYCL_V3(m1, m2, m3), SYCL_V2(m4, m5)
#define SYCL_V6(m1, m2, m3, m4, m5, m6)				SYCL_V3(m1, m2, m3), SYCL_V3(m4, m5, m6)
#define SYCL_V7(m1, m2, m3, m4, m5, m6, m7)			SYCL_V4(m1, m2, m3, m4), SYCL_V3(m5, m6, m7)
#define SYCL_V8(m1, m2, m3, m4, m5, m6, m7, m8)		SYCL_V4(m1, m2, m3, m4), SYCL_V4(m5, m6, m7, m8)
#define SYCL_V9(m1, m2, m3, m4, m5, m6, m7, m8, m9)	SYCL_V5(m1, m2, m3, m4, m5), SYCL_V4(m6, m7, m8, m9)

template <typename dataT>
struct vec_members<dataT, 2> : vec_members<dataT, 1> {
	vec_base<dataT, 1> x, y, s0, s1;
	vec_base<dataT, 1> lo, hi;
	vec_base<dataT, 2> xx, xy, yx, yy;
	vec_base<dataT, 2> s00, s01, s10, s11;
	vec_members(string_class name)
		:	SYCL_V4(x, y, s0, s1),
			SYCL_V2(lo, hi),
			SYCL_V4(xx, xy, yx, yy),
			SYCL_V4(s00, s01, s10, s11) {}
};

template <typename dataT>
struct vec_members<dataT, 3> : vec_members<dataT, 2> {
	vec_base<dataT, 1> z, s2;
	vec_base<dataT, 2> lo, hi;
	vec_base<dataT, 2> xz, yz, zx, zy, zz;
	vec_base<dataT, 2> s02, s12, s20, s21, s22;
	vec_base<dataT, 3>	xxx, xxy, xxz,
						xyx, xyy, xyz,
						xzx, xzy, xzz,
						yxx, yxy, yxz,
						yyx, yyy, yyz,
						yzx, yzy, yzz,
						zxx, zxy, zxz,
						zyx, zyy, zyz,
						zzx, zzy, zzz;
	vec_base<dataT, 3>	s000, s001, s002,
						s010, s011, s012,
						s020, s021, s022,
						s100, s101, s102,
						s110, s111, s112,
						s120, s121, s122,
						s200, s201, s202,
						s210, s211, s212,
						s220, s221, s222;
	vec_members(string_class name)
		:	vec_members<dataT, 2>(name),
			SYCL_V2(z, s2),
			SYCL_V2(lo, hi),
			SYCL_V5(xz, yz, zx, zy, zz),
			SYCL_V5(s02, s12, s20, s21, s22),
			SYCL_V9(xxx, xxy, xxz,
					xyx, xyy, xyz,
					xzx, xzy, xzz),
			SYCL_V9(yxx, yxy, yxz,
					yyx, yyy, yyz,
					yzx, yzy, yzz),
			SYCL_V9(zxx, zxy, zxz,
					zyx, zyy, zyz,
					zzx, zzy, zzz),
			SYCL_V9(s000, s001, s002,
					s010, s011, s012,
					s020, s021, s022),
			SYCL_V9(s100, s101, s102,
					s110, s111, s112,
					s120, s121, s122),
			SYCL_V9(s200, s201, s202,
					s210, s211, s212,
					s220, s221, s222) {}
};

// TODO: Higher dimensions

#undef SYCL_V
#undef SYCL_V2
#undef SYCL_V3
#undef SYCL_V4
#undef SYCL_V5
#undef SYCL_V6
#undef SYCL_V7
#undef SYCL_V8
#undef SYCL_V9

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

	template <int num = numElements>
	vec_base(data_ref x, data_ref y, SYCL_ENABLE_IF_DIM(2))
		: counter(), data_ref(generate_name()) {
		detail::kernel_add(type_name() + ' ' + name + " = (" + x.name + ", " + y.name + ")");
	}

	template <int num = numElements>
	vec_base(data_ref x, data_ref y, data_ref z, SYCL_ENABLE_IF_DIM(3))
		: counter(), data_ref(generate_name()) {
		detail::kernel_add(type_name() + ' ' + name + " = (" + x.name + ", " + y.name + ", " + z.name + ")");
	}

	// TODO: Swizzle methods
	//swizzled_vec<T, out_dims> swizzle<int s1, ...>();
#ifdef SYCL_SIMPLE_SWIZZLES
	swizzled_vec<T, 4> xyzw();
	...
#endif // #ifdef SYCL_SIMPLE_SWIZZLES
};

} // namespace detail

#if MSVC_LOW
#define SYCL_VEC_INHERIT_CONSTRUCTORS(type)							\
	type()															\
		: Base() {}													\
	template <class T>												\
	type(T n)														\
		: Base(n) {}												\
	template <int num = numElements>								\
	type(data_ref x, data_ref y, SYCL_ENABLE_IF_DIM(2))				\
		: Base(x, y) {}												\
	template <int num = numElements>								\
	type(data_ref x, data_ref y, data_ref z, SYCL_ENABLE_IF_DIM(3))	\
		: Base(x, y, z) {}
#else
#define SYCL_VEC_INHERIT_CONSTRUCTORS(type)	\
	using Base::Base;
#endif

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
	
	template <int num = numElements>
	vec(data_ref x, data_ref y, SYCL_ENABLE_IF_DIM(2))
		: Base(x, y), Members(name) {}
	template <int num = numElements>
	vec(data_ref x, data_ref y, data_ref z, SYCL_ENABLE_IF_DIM(3))
		: Base(x, y, z), Members(name) {}
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

#undef SYCL_ENABLE_IF_DIM

#undef SYCL_ADD_SIGNED_vec
#undef SYCL_ADD_UNSIGNED_vec
#undef SYCL_VEC_INHERIT_ASSIGNMENTS
#undef SYCL_VEC_INHERIT_CONSTRUCTORS
#undef SYCL_VEC_SIGNED
#undef SYCL_VEC_UNSIGNED

} // namespace sycl
} // namespace cl
