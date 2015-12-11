#pragma once

namespace cl {
namespace sycl {

// Forward declaration
template <typename dataT, int numElements>
class vec;

namespace detail {
namespace vectors {

// Forward declaration
template <typename dataT, int numElements>
class base;


template <typename dataT>
using single_member = base<dataT, 1>;

template <typename dataT, int numElements>
struct members {
	members(base<dataT, numElements>* parent) {}
};


#define SYCL_V(member)								member(parent->name + "." #member)
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

	members(base<dataT, 2>* parent)
	:	SYCL_V2(x, y),
		SYCL_R2(x, lo, s0),
		SYCL_R2(y, hi, s1) {}
};

#define SYCL_MEMBERS_3()	\
SYCL_V2(lo, hi),			\
SYCL_R2(lo.x, x, s0),		\
SYCL_R2(lo.y, y, s1),		\
SYCL_R2(hi.x, z, s2)


#define SYCL_SWIZZLE_3_SREFS()			\
base<dataT, 3>	&s000, &s001, &s002,	\
				&s010, &s011,			\
				&s020, &s021, &s022,	\
				&s100, &s101, &s102,	\
				&s110, &s111, &s112,	\
				&s120, &s121, &s122,	\
				&s200, &s201, &s202,	\
				&s210, &s211, &s212,	\
				&s220, &s221, &s222;

#define SYCL_SWIZZLE_3_DATA()	\
base<dataT, 3>	xxx, xxy, xxz,	\
				xyx, xyy,		\
				xzx, xzy, xzz,	\
				yxx, yxy, yxz,	\
				yyx, yyy, yyz,	\
				yzx, yzy, yzz,	\
				zxx, zxy, zxz,	\
				zyx, zyy, zyz,	\
				zzx, zzy, zzz;	\
	SYCL_SWIZZLE_3_SREFS()

#define SYCL_SWIZZLE_3_SET_VALUES()			\
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
	SYCL_SWIZZLE_3_DATA();

	members(base<dataT, 3>* parent)
	:	SYCL_MEMBERS_3(),
		SYCL_R2(*parent, xyz, s012),
		SYCL_SWIZZLE_3_SET_VALUES() {}
};

template <typename dataT>
struct members<dataT, 4> {
	vec<dataT, 2> lo, hi;
	single_member<dataT> &x, &y, &z, &w;
	single_member<dataT> &s0, &s1, &s2, &s3;
	vec<dataT, 3> xyz, &s012;
	SYCL_SWIZZLE_3_DATA();
	vec<dataT, 3> yzw;

	members(base<dataT, 4>* parent)
	:	SYCL_MEMBERS_3(),
		SYCL_R2(hi.y, w, s3),
		SYCL_V(xyz), s012(xyz),
		SYCL_SWIZZLE_3_SET_VALUES(),
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


#define SYCL_SWIZZLE_3_VALUE_REFS()	\
base<dataT, 3>	&xxx, &xxy, &xxz,	\
				&xyx, &xyy,			\
				&xzx, &xzy, &xzz,	\
				&yxx, &yxy, &yxz,	\
				&yyx, &yyy, &yyz,	\
				&yzx, &yzy, &yzz,	\
				&zxx, &zxy, &zxz,	\
				&zyx, &zyy, &zyz,	\
				&zzx, &zzy, &zzz;	\
	SYCL_SWIZZLE_3_SREFS()

#define SYCL_SWIZZLE_3_REFS_9(pf, vf, sf)	\
SYCL_R2_LO(pf, vf##xx, sf##00),				\
SYCL_R2_LO(pf, vf##xy, sf##01),				\
SYCL_R2_LO(pf, vf##xz, sf##02),				\
SYCL_R2_LO(pf, vf##yx, sf##10),				\
SYCL_R2_LO(pf, vf##yy, sf##11),				\
SYCL_R2_LO(pf, vf##yz, sf##12),				\
SYCL_R2_LO(pf, vf##zx, sf##20),				\
SYCL_R2_LO(pf, vf##zy, sf##21),				\
SYCL_R2_LO(pf, vf##zz, sf##22)

#define SYCL_SWIZZLE_3_SET_REFS(pf)	\
SYCL_SWIZZLE_3_REFS_9(pf, x, s0),	\
SYCL_SWIZZLE_3_REFS_9(pf, y, s1),	\
SYCL_SWIZZLE_3_REFS_9(pf, z, s2)

template <typename dataT>
struct members<dataT, 8> {
	vec<dataT, 4> lo, hi;
	single_member<dataT> &x, &y, &z, &w;
	single_member<dataT> &s0, &s1, &s2, &s3, &s4, &s5, &s6, &s7;
	vec<dataT, 3> &xyz, &s012;
	SYCL_SWIZZLE_3_VALUE_REFS();
	vec<dataT, 3> &yzw;

	members(base<dataT, 8>* parent)
	:	SYCL_MEMBERS_8(this->),
		SYCL_SWIZZLE_3_SET_REFS(this->),
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
	SYCL_SWIZZLE_3_VALUE_REFS();
	vec<dataT, 3> &yzw;

	members(base<dataT, 16>* parent)
	:	SYCL_MEMBERS_8(lo.),
		s8(hi.x), s9(hi.y), sa(hi.z), sb(hi.w),
		sc(hi.hi.x), sd(hi.hi.y), se(hi.hi.z), sf(hi.hi.w),
		SYCL_SWIZZLE_3_SET_REFS(lo.),
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

#undef SYCL_SWIZZLE_3_DATA
#undef SYCL_SWIZZLE_3_SREFS
#undef SYCL_SWIZZLE_3_VALUE_REFS
#undef SYCL_SWIZZLE_3_SET_VALUES
#undef SYCL_SWIZZLE_3_SET_REFS
#undef SYCL_SWIZZLE_3_REFS_9

} // namespace vectors
} // namespace detail

} // namespace sycl
} // namespace cl
