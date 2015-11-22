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


template <typename dataT>
using single_member = base<dataT, 1>;

template <typename dataT, int numElements>
struct members {
	members(base<dataT, numElements>* parent, string_class name = "") {}
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

#define SYCL_R2(org, r1, r2)	r1(org), r2(org)
#define SYCL_R2_LO(pf, r1, r2)	r1(pf lo.r1), r2(pf lo.r1)

// TODO: All members

template <typename dataT>
struct members<dataT, 2> {
	single_member<dataT> x, y;
	single_member<dataT> &lo, &hi;
	single_member<dataT> &s0, &s1;

	members(base<dataT, 2>* parent, string_class name)
	:	SYCL_V2(x, y),
		SYCL_R2(x, lo, s0),
		SYCL_R2(y, hi, s1) {}
};

#define SYCL_MEMBERS_3()	\
SYCL_V2(lo, hi),			\
SYCL_R2(lo.x, x, s0),		\
SYCL_R2(lo.y, y, s1),		\
SYCL_R2(hi.x, z, s2)

#define SYCL_SWIZZLE_3(pf)				\
base<dataT, 3>	pf xxx, pf xxy, pf xxz,	\
				pf xyx, pf xyy,			\
				pf xzx, pf xzy, pf xzz,	\
				pf yxx, pf yxy, pf yxz,	\
				pf yyx, pf yyy, pf yyz,	\
				pf yzx, pf yzy, pf yzz,	\
				pf zxx, pf zxy, pf zxz,	\
				pf zyx, pf zyy, pf zyz,	\
				pf zzx, pf zzy, pf zzz;	\
base<dataT, 3>	&s000, &s001, &s002,	\
				&s010, &s011,			\
				&s020, &s021, &s022,	\
				&s100, &s101, &s102,	\
				&s110, &s111, &s112,	\
				&s120, &s121, &s122,	\
				&s200, &s201, &s202,	\
				&s210, &s211, &s212,	\
				&s220, &s221, &s222;

#define SYCL_SWIZZLE_3_VALUES()				\
SYCL_V8(xxx, xxy, xxz,						\
		xyx, xyy,							\
		xzx, xzy, xzz),						\
SYCL_V9(yxx, yxy, yxz,						\
		yyx, yyy, yyz,						\
		yzx, yzy, yzz),						\
SYCL_V9(zxx, zxy, zxz,						\
		zyx, zyy, zyz,						\
		zzx, zzy, zzz),						\
		s000(xxx), s001(xxy), s002(xxz),	\
		s010(xyx), s011(xyy),				\
		s020(xzx), s021(xzy), s022(xzz),	\
		s100(yxx), s101(yxy), s102(yxz),	\
		s110(yyx), s111(yyy), s112(yyz),	\
		s120(yzx), s121(yzy), s122(yzz),	\
		s200(zxx), s201(zxy), s202(zxz),	\
		s210(zyx), s211(zyy), s212(zyz),	\
		s220(zzx), s221(zzy), s222(zzz)

template <typename dataT>
struct members<dataT, 3> {
	vec<dataT, 2> lo, hi;
	single_member<dataT> &x, &y, &z;
	single_member<dataT> &s0, &s1, &s2;
	vec<dataT, 3> &xyz, &s012;
	SYCL_SWIZZLE_3();

	members(base<dataT, 3>* parent, string_class name)
	:	SYCL_MEMBERS_3(),
		SYCL_R2(*parent, xyz, s012),
		SYCL_SWIZZLE_3_VALUES() {}
};

template <typename dataT>
struct members<dataT, 4> {
	vec<dataT, 2> lo, hi;
	single_member<dataT> &x, &y, &z, &w;
	single_member<dataT> &s0, &s1, &s2, &s3;
	vec<dataT, 3> xyz, &s012;
	SYCL_SWIZZLE_3();
	vec<dataT, 3> yzw;

	members(base<dataT, 4>* parent, string_class name)
	:	SYCL_MEMBERS_3(),
		SYCL_R2(hi.y, w, s3),
		SYCL_V(xyz), s012(xyz),
		SYCL_SWIZZLE_3_VALUES(),
		SYCL_V(yzw) {}
};

#define SYCL_MEMBERS_8(pf)	\
SYCL_V2(lo, hi),			\
SYCL_R2_LO(pf, x, s0),		\
SYCL_R2_LO(pf, y, s1),		\
SYCL_R2_LO(pf, z, s2),		\
SYCL_R2_LO(pf, w, s3),		\
s4(pf hi.x),				\
s5(pf hi.y),				\
s6(pf hi.z),				\
s7(pf hi.w)

#define SYCL_SWIZZLE_3_REFS_9(pf, vf, sf)	\
SYCL_R2_LO(pf, vf##xx, sf##00),				\
SYCL_R2_LO(pf, vf##xy, sf##01),				\
SYCL_R2_LO(pf, vf##xz, sf##02),				\
SYCL_R2_LO(pf, vf##yx, sf##10),				\
SYCL_R2_LO(pf, vf##yy, sf##11),				\
SYCL_R2_LO(pf, vf##yz, sf##12),				\
SYCL_R2_LO(pf, vf##zx, sf##21),				\
SYCL_R2_LO(pf, vf##zy, sf##21),				\
SYCL_R2_LO(pf, vf##zz, sf##22)

#define SYCL_SWIZZLE_3_REFS(pf)		\
SYCL_SWIZZLE_3_REFS_9(pf, x, s0),	\
SYCL_SWIZZLE_3_REFS_9(pf, y, s1),	\
SYCL_SWIZZLE_3_REFS_9(pf, z, s2)

template <typename dataT>
struct members<dataT, 8> {
	vec<dataT, 4> lo, hi;
	single_member<dataT> &x, &y, &z, &w;
	single_member<dataT> &s0, &s1, &s2, &s3, &s4, &s5, &s6, &s7;
	vec<dataT, 3> &xyz, &s012;
	SYCL_SWIZZLE_3(&);
	vec<dataT, 3> &yzw;

	members(base<dataT, 8>* parent, string_class name)
	:	SYCL_MEMBERS_8(this->),
		SYCL_SWIZZLE_3_REFS(this->),
		yzw(lo.yzw) {}
};

template <typename dataT>
struct members<dataT, 16> {
	vec<dataT, 8> lo, hi;
	single_member<dataT> &x, &y, &z, &w;
	single_member<dataT>	&s0, &s1, &s2, &s3,
							&s4, &s5, &s6, &s7,
							&s8, &s9, &sa, &sb,
							&sc, &sd, &se, &sf;
	vec<dataT, 3> &xyz, &s012;
	SYCL_SWIZZLE_3(&);
	vec<dataT, 3> &yzw;

	members(base<dataT, 16>* parent, string_class name)
	:	SYCL_MEMBERS_8(lo.),
		SYCL_SWIZZLE_3_REFS(lo.),
		yzw(lo.yzw) {}
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
#undef SYCL_R2_LO
#undef SYCL_MEMBERS_3
#undef SYCL_MEMBERS_8

#undef SYCL_SWIZZLE_3
#undef SYCL_SWIZZLE_3_VALUES
#undef SYCL_SWIZZLE_3_REFS
#undef SYCL_SWIZZLE_3_REFS_9


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
