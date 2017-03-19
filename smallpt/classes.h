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
struct Vec_detail {
  type x, y, z;
  Vec_detail(const type& x = 0, const type& y = 0, const type& z = 0) {
    this->x = x;
    this->y = y;
    this->z = z;
  }
  Vec_detail operator+(const Vec_detail& b) const {
    return Vec_detail(this->x + b.x, this->y + b.y, this->z + b.z);
  }
  Vec_detail operator-(const Vec_detail& b) const {
    return Vec_detail(this->x - b.x, this->y - b.y, this->z - b.z);
  }
  Vec_detail operator*(const type& b) const {
    return Vec_detail(this->x * b, this->y * b, this->z * b);
  }
  Vec_detail mult(const Vec_detail& b) const {
    return Vec_detail(this->x * b.x, this->y * b.y, this->z * b.z);
  }
  Vec_detail& norm() {
    return *this = *this * (1 / sqrt_f(this->x * this->x + this->y * this->y +
                                       this->z * this->z));
  }
  type dot(const Vec_detail& b) const {
    return this->x * b.x + this->y * b.y + this->z * b.z;
  }  // cross:
  Vec_detail operator%(Vec_detail& b) {
    return Vec_detail(this->y * b.z - this->z * b.y,
                      this->z * b.x - this->x * b.z,
                      this->x * b.y - this->y * b.x);
  }
};
template <class type>
struct Ray_detail {
  Vec_detail<type> o, d;
  Ray_detail(Vec_detail<type> o, Vec_detail<type> d) : o(o), d(d) {}
};
enum Refl_t { DIFF, SPEC, REFR };  // material types, used in radiance()
template <class type, int shouldModifySampleRate = 1>
struct Sphere_detail {
  type rad;                  // radius
  Vec_detail<type> p, e, c;  // position, emission, color
  Refl_t refl;               // reflection type (DIFFuse, SPECular, REFRactive)
  Sphere_detail(const type& rad, Vec_detail<type> p, Vec_detail<type> e,
                Vec_detail<type> c, Refl_t refl)
      : rad(rad), p(p), e(e), c(c), refl(refl) {}
  type intersect(
      const Ray_detail<type>& r) const {  // returns distance, 0 if nohit
    Vec_detail<type> op =
        p - r.o;  // Solve t^2*d.d + 2*t*(o-p).d + (o-p).(o-p)-R^2 = 0
    type t, eps = static_cast<type>(1e-4 * shouldModifySampleRate),
            b = op.dot(r.d), det = b * b - op.dot(op) + rad * rad;
    if (det < 0) {
      return 0;
    } else {
      det = sqrt_f(det);
    }
    return (t = b - det) > eps ? t : ((t = b + det) > eps ? t : 0);
  }
};

#undef sqrt_f
