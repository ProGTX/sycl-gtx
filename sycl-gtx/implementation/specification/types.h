#pragma once

// 3.7 Data Types

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

#define SYCL_ENABLE_IF_DIM(dim)	\
typename std::enable_if<num == dim>::type* = nullptr

// Forward declaration
template <typename dataT, int numElements>
class base;

template <typename dataT, int numElements>
struct members {
	members(string_class name = "") {}
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

#define SYCL_R2(org, m1, m2)	m1(org), m2(org)

template <typename dataT>
struct members<dataT, 2> : members<dataT, 1> {
	base<dataT, 1> x, y;
	base<dataT, 1> &s0, &s1, &lo, &hi;
	base<dataT, 2> xx, xy, yx, yy;
	base<dataT, 2> &s00, &s01, &s10, &s11;
	members(string_class name)
		:	SYCL_V2(x, y),
			SYCL_R2(x, s0, lo),
			SYCL_R2(y, s1, hi),
			SYCL_V4(xx, xy, yx, yy),
			s00(xx), s01(xy), s10(yx), s11(yy) {}
};

template <typename dataT>
struct members<dataT, 3> : members<dataT, 2> {
	base<dataT, 1> z;
	base<dataT, 1> &s2;
	members<dataT, 2> lo, hi;
	base<dataT, 2> xz, yz, zx, zy, zz;
	base<dataT, 2> &s02, &s12, &s20, &s21, &s22;
	base<dataT, 3>	xxx, xxy, xxz,
						xyx, xyy, xyz,
						xzx, xzy, xzz,
						yxx, yxy, yxz,
						yyx, yyy, yyz,
						yzx, yzy, yzz,
						zxx, zxy, zxz,
						zyx, zyy, zyz,
						zzx, zzy, zzz;
	base<dataT, 3>	&s000, &s001, &s002,
						&s010, &s011, &s012,
						&s020, &s021, &s022,
						&s100, &s101, &s102,
						&s110, &s111, &s112,
						&s120, &s121, &s122,
						&s200, &s201, &s202,
						&s210, &s211, &s212,
						&s220, &s221, &s222;
	members(string_class name)
		:	members<dataT, 2>(name),
			SYCL_V(z), s2(z),
			SYCL_V2(lo, hi),
			SYCL_V5(xz, yz, zx, zy, zz),
			s02(xz), s12(yz), s20(zx), s21(zy), s22(zz),
			SYCL_V9(xxx, xxy, xxz,
					xyx, xyy, xyz,
					xzx, xzy, xzz),
			SYCL_V9(yxx, yxy, yxz,
					yyx, yyy, yyz,
					yzx, yzy, yzz),
			SYCL_V9(zxx, zxy, zxz,
					zyx, zyy, zyz,
					zzx, zzy, zzz),
			s000(xxx), s001(xxy), s002(xxz),
			s010(xyx), s011(xyy), s012(xyz),
			s020(xzx), s021(xzy), s022(xzz),
			s100(yxx), s101(yxy), s102(yxz),
			s110(yyx), s111(yyy), s112(yyz),
			s120(yzx), s121(yzy), s122(yzz),
			s200(zxx), s201(zxy), s202(zxz),
			s210(zyx), s211(zyy), s212(zyz),
			s220(zzx), s221(zzy), s222(zzz) {}
};

// TODO: All members

template <typename dataT>
struct members<dataT, 4> : members<dataT, 3> {
	base<dataT, 1> w;
	base<dataT, 1> &s3;
	base<dataT, 3> yzw;
	base<dataT, 3> &s123;
	base<dataT, 4> xyzw;
	base<dataT, 4> &s0123;

	members(string_class name)
		:	members<dataT, 3>(name),
			SYCL_V(w), s3(w),
			SYCL_V(yzw), s123(yzw),
			SYCL_V(xyzw), s0123(xyzw) {}
};

template <typename dataT>
struct members<dataT, 8> : members<dataT, 4> {
	members<dataT, 4> lo, hi;

	members(string_class name)
		:	members<dataT, 4>(name),
			SYCL_V2(lo, hi) {}
};

template <typename dataT>
struct members<dataT, 16> : members<dataT, 8> {
	members<dataT, 8> lo, hi;

	members(string_class name)
		:	members<dataT, 8>(name),
			SYCL_V2(lo, hi) {}
};


#undef SYCL_V
#undef SYCL_V2
#undef SYCL_V3
#undef SYCL_V4
#undef SYCL_V5
#undef SYCL_V6
#undef SYCL_V7
#undef SYCL_V8
#undef SYCL_V9

#undef SYCL_R2


template <typename dataT, int numElements>
class base : protected counter<base<dataT, numElements>>, public data_ref {
private:
	template <typename dataT, int numElements>
	friend struct members;
	template <typename DataType>
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

	base(string_class name)
		: data_ref(name) {}

public:
	base()
		: data_ref(generate_name()) {
		kernel_add(this_name());
	}

	template <class T>
	base(T n)
		: data_ref(generate_name()) {
		kernel_add(this_name() + " = " + get_name(n));
	}

	template <int num = numElements>
	base(data_ref x, data_ref y, SYCL_ENABLE_IF_DIM(2))
		: data_ref(generate_name()) {
		kernel_add(this_name() + " = (" + x.name + ", " + y.name + ')');
	}

	template <int num = numElements>
	base(data_ref x, data_ref y, data_ref z, SYCL_ENABLE_IF_DIM(3))
		: data_ref(generate_name()) {
		kernel_add(this_name() + " = (" + x.name + ", " + y.name + ", " + z.name + ')');
	}

	operator vec<dataT, numElements>&() {
		return *reinterpret_cast<vec<dataT, numElements>*>(this);
	}

	// TODO: Swizzle methods
	//swizzled_vec<T, out_dims> swizzle<int s1, ...>();
#ifdef SYCL_SIMPLE_SWIZZLES
	swizzled_vec<T, 4> xyzw();
	...
#endif // #ifdef SYCL_SIMPLE_SWIZZLES
};

template <typename dataT, int numElements>
struct data;

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
	using Base = detail::vectors::base<dataT, numElements>;
	using Members = detail::vectors::members<dataT, numElements>;
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

	template <class T>
	vec& operator=(T n) {
		data_ref::operator=(n);
		return *this;
	}
};


// 3.9.1 Description of the built-in types available for SYCL host and device

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
