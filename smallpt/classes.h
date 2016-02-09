#pragma once
// smallpt, a Path Tracer by Kevin Beason, 2008
//
// Modified by Peter Žužek
// For the original code, see github.com/munificient/smallpt
// For the original license, see smallpt.LICENSE.txt

#ifndef sqrt_f
#include <math.h>
#define sqrt_f sqrt
#endif

template <class type>
struct Vec_ {
  type x, y, z;
  Vec_(const type& x_ = 0, const type& y_ = 0, const type& z_ = 0) { x = x_; y = y_; z = z_; }
  Vec_ operator+(const Vec_ &b) const { return Vec_(x+b.x,y+b.y,z+b.z); }
  Vec_ operator-(const Vec_ &b) const { return Vec_(x-b.x,y-b.y,z-b.z); }
  Vec_ operator*(const type& b) const { return Vec_(x*b, y*b, z*b); }
  Vec_ mult(const Vec_ &b) const { return Vec_(x*b.x,y*b.y,z*b.z); }
  Vec_& norm(){ return *this = *this * (1/sqrt(x*x+y*y+z*z)); }
  type dot(const Vec_ &b) const { return x*b.x+y*b.y+z*b.z; } // cross:
  Vec_ operator%(Vec_&b){return Vec_(y*b.z-z*b.y,z*b.x-x*b.z,x*b.y-y*b.x);}
};
template <class type>
struct Ray_ { Vec_<type> o, d; Ray_(Vec_<type> o_, Vec_<type> d_) : o(o_), d(d_) {} };
enum Refl_t { DIFF, SPEC, REFR };  // material types, used in radiance()
template <class type>
struct Sphere_ {
  type rad;       // radius
  Vec_<type> p, e, c;      // position, emission, color
  Refl_t refl;      // reflection type (DIFFuse, SPECular, REFRactive)
  Sphere_(const type& rad_, Vec_<type> p_, Vec_<type> e_, Vec_<type> c_, Refl_t refl_) :
    rad(rad_), p(p_), e(e_), c(c_), refl(refl_) {}
  type intersect(const Ray_<type> &r) const { // returns distance, 0 if nohit
    Vec_<type> op = p-r.o; // Solve t^2*d.d + 2*t*(o-p).d + (o-p).(o-p)-R^2 = 0
    type t, eps=(type)1e-4, b=op.dot(r.d), det=b*b-op.dot(op)+rad*rad;
    if (det<0) return 0; else det=sqrt(det);
    return (t=b-det)>eps ? t : ((t=b+det)>eps ? t : 0);
  }
};

#undef sqrt_f
