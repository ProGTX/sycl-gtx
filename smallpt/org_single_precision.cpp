// smallpt, a Path Tracer by Kevin Beason, 2008
//
// Modified by Peter Žužek
// For the original code, see github.com/munificient/smallpt
// For the original license, see smallpt.LICENSE.txt

#include <math.h>   // smallpt, a Path Tracer by Kevin Beason, 2008
#include <stdlib.h> // Make : g++ -O3 -fopenmp smallpt.cpp -o smallpt
#include <stdio.h>  //        Remove "-fopenmp" for g++ version < 4.2
// Usage: time ./smallpt 5000 && xv image.ppm
// position, also color (r,g,b)

#include "classes.h"
#include "win.h"
#include <vector>

using Vec = Vec_<double>;
using Ray = Ray_<double>;
using Sphere = Sphere_<double>;

namespace org_sp {
struct Vec {
  float x, y, z;
  Vec(float x_ = 0, float y_ = 0, float z_ = 0) { x = x_; y = y_; z = z_; } 
  Vec(const ::Vec& v) : Vec((float)v.x, (float)v.y, (float)v.z) {}
  Vec operator+(const Vec &b) const { return Vec(x + b.x, y + b.y, z + b.z); }
  Vec operator-(const Vec &b) const { return Vec(x - b.x, y - b.y, z - b.z); }
  Vec operator*(float b) const { return Vec(x*b, y*b, z*b); }
  Vec mult(const Vec &b) const { return Vec(x*b.x, y*b.y, z*b.z); }
  Vec& norm() { return *this = *this * (1 / sqrt(x*x + y*y + z*z)); }
  float dot(const Vec &b) const { return x*b.x + y*b.y + z*b.z; } // cross:
  Vec operator%(Vec&b) { return Vec(y*b.z - z*b.y, z*b.x - x*b.z, x*b.y - y*b.x); }
};
struct Ray {
  Vec o, d;
  Ray(Vec o_, Vec d_) : o(o_), d(d_) {}
  Ray(const ::Ray& r) : o(r.o), d(r.d) {}
};
enum Refl_t { DIFF, SPEC, REFR };  // material types, used in radiance()
struct Sphere {
  float rad;       // radius
  Vec p, e, c;      // position, emission, color
  Refl_t refl;      // reflection type (DIFFuse, SPECular, REFRactive)
  Sphere(float rad_, Vec p_, Vec e_, Vec c_, Refl_t refl_) :
    rad(rad_), p(p_), e(e_), c(c_), refl(refl_) {}
  float intersect(const Ray &r) const { // returns distance, 0 if nohit
    Vec op = p - r.o; // Solve t^2*d.d + 2*t*(o-p).d + (o-p).(o-p)-R^2 = 0
    float t, eps = 1e-2f, b = op.dot(r.d), det = b*b - op.dot(op) + rad*rad;
    if(det<0) return 0; else det = sqrt(det);
    return (t = b - det)>eps ? t : ((t = b + det)>eps ? t : 0);
  }
};
Sphere spheres[] = {//Scene: radius, position, emission, color, material
  Sphere(1e4f, Vec(1e4f + 1, 40.8f, 81.6f), Vec(), Vec(.75f, .25f, .25f), DIFF),//Left
  Sphere(1e4f, Vec(-1e4f + 99, 40.8f, 81.6f), Vec(), Vec(.25f, .25f, .75f), DIFF),//Rg
  Sphere(1e4f, Vec(50, 40.8f, 1e4f), Vec(), Vec(.75f, .75f, .75f), DIFF),//Back
  Sphere(1e4f, Vec(50, 40.8f, -1e4f + 170), Vec(), Vec(), DIFF),//Front
  Sphere(1e4f, Vec(50, 1e4f, 81.6f), Vec(), Vec(.75f, .75f, .75f), DIFF),//Bottom
  Sphere(1e4f, Vec(50, -1e4f + 81.6f, 81.6f), Vec(), Vec(.75f, .75f, .75f), DIFF),//Tp
  Sphere(16.5f, Vec(27, 16.5f, 47), Vec(), Vec(1, 1, 1)*.999f, SPEC),//Mirr
  Sphere(16.5f, Vec(73, 16.5f, 78), Vec(), Vec(1, 1, 1)*.999f, REFR),//Glas
  Sphere(600, Vec(50, 681.6f - .27f, 81.6f), Vec(12, 12, 12), Vec(), DIFF) //Lite
};
inline float clamp(float x) { return x < 0 ? 0 : x>1 ? 1 : x; }
inline bool intersect(const Ray &r, float &t, int &id) {
  float n = sizeof(spheres) / sizeof(Sphere), d, inf = t = 1e20f; 
  for(int i = int(n); i--;) {
    if((d = spheres[i].intersect(r)) && d < t) { t = d; id = i; }
  }
  return t<inf;
}
Vec radiance(const Ray &r, int depth, unsigned short *Xi) {
  float t;                               // distance to intersection
  int id = 0;                            // id of intersected object
  if(!intersect(r, t, id)) return Vec(); // if miss, return black
  const Sphere &obj = spheres[id];       // the hit object
  Vec x = r.o + r.d*t;
  Vec n = (x - obj.p).norm(), nl = n.dot(r.d) < 0 ? n : n*-1, f = obj.c;
  float p = f.x > f.y && f.x>f.z ? f.x : f.y > f.z ? f.y : f.z; // max refl
  if(depth > 255) return obj.e;
  if(++depth > 5) {
    if(get_random(Xi) < p) { f = f*(1 / p); } else { return obj.e; }
  } //R.R.
  if(obj.refl == DIFF) {                  // Ideal DIFFUSE reflection
    float r1 = (float)(2 * M_PI*get_random(Xi));
    float r2 = (float)get_random(Xi), r2s = sqrt(r2);
    Vec w = nl, u = ((fabs(w.x) > .1 ? Vec(0, 1) : Vec(1)) % w).norm(), v = w%u;
    Vec d = (u*cos(r1)*r2s + v*sin(r1)*r2s + w*sqrt(1 - r2)).norm();
    return obj.e + f.mult(radiance(Ray(x, d), depth, Xi));
  }
  else if(obj.refl == SPEC)            // Ideal SPECULAR reflection
    return obj.e + f.mult(radiance(Ray(x, r.d - n * 2 * n.dot(r.d)), depth, Xi));
  Ray reflRay(x, r.d - n * 2 * n.dot(r.d));     // Ideal dielectric REFRACTION
  bool into = n.dot(nl) > 0;                // Ray from outside going in?
  float nc = 1, nt = 1.5f, nnt = into ? nc / nt : nt / nc, ddn = r.d.dot(nl), cos2t;
  if((cos2t = 1 - nnt*nnt*(1 - ddn*ddn)) < 0)    // Total internal reflection
    return obj.e + f.mult(radiance(reflRay, depth, Xi));
  Vec tdir = (r.d*nnt - n*((into ? 1 : -1)*(ddn*nnt + sqrt(cos2t)))).norm();
  float a = nt - nc;
  float b = nt + nc, R0 = a*a / (b*b), c = 1 - (into ? -ddn : tdir.dot(n));
  float Re = R0 + (1 - R0)*c*c*c*c*c;
  float Tr = 1 - Re, P = .25f + .5f*Re, RP = Re / P, TP = Tr / (1 - P);
  return obj.e + f.mult(depth > 2 ? (get_random(Xi) < P ?   // Russian roulette
    radiance(reflRay, depth, Xi)*RP : radiance(Ray(x, tdir), depth, Xi)*TP) :
    radiance(reflRay, depth, Xi)*Re + radiance(Ray(x, tdir), depth, Xi)*Tr);
}
inline void compute_inner(
  int y, int w, int h, int samps, Ray &cam, Vec &cx, Vec &cy, Vec &r, Vec *c
) {
  //fprintf(stderr, "\rRendering (%d spp) %5.2f%%", samps * 4, 100.*y / (h - 1));
  // Loop cols
  for(unsigned short x = 0, Xi[3] = { 0, 0, (unsigned short)(y*y*y) }; x < w; x++)   
    for(int sy = 0, i = (h - y - 1)*w + x; sy < 2; sy++)     // 2x2 subpixel rows
      for(int sx = 0; sx < 2; sx++, r = Vec()) {        // 2x2 subpixel cols
        for(int s = 0; s < samps; s++) {
          float r1 =
            (float)(2 * get_random(Xi)), dx = r1 < 1 ? sqrt(r1) - 1 : 1 - sqrt(2-r1);
          float r2 =
            (float)(2 * get_random(Xi)), dy = r2 < 1 ? sqrt(r2) - 1 : 1 - sqrt(2-r2);
          Vec d = cx*(((sx + .5f + dx) / 2 + x) / w - .5f) +
            cy*(((sy + .5f + dy) / 2 + y) / h - .5f) + cam.d;
          r = r + radiance(Ray(cam.o + d * 140, d.norm()), 0, Xi)*(1.f / samps);
        } // Camera rays are pushed ^^^^^ forward to start in interior
        c[i] = c[i] + Vec(clamp(r.x), clamp(r.y), clamp(r.z))*.25f;
      }
}
std::vector<org_sp::Vec> get_c(int w, int h, ::Vec* c_) {
  std::vector<org_sp::Vec> c;
  c.reserve(w*h);
  for(int y = 0; y < h; ++y) {
    for(int x = 0; x < w; ++x) {
      int i = y*w + x;
      c.emplace_back(c_[i]);
    }
  }
  return c;
}
void assign_c(int w, int h, std::vector<org_sp::Vec>& c, ::Vec* c_) {
  for(int y = 0; y < h; ++y) {
    for(int x = 0; x < w; ++x) {
      int i = y*w + x;
      c_[i].x = c[i].x;
      c_[i].y = c[i].y;
      c_[i].z = c[i].z;
    }
  }
}
} // namespace org_sp
void compute_org_sp(
  void*, int w, int h, int samps, Ray cam_, Vec cx_, Vec cy_, Vec r_, Vec *c_) {
  org_sp::Ray cam(cam_);
  org_sp::Vec cx(cx_);
  org_sp::Vec cy(cy_);
  org_sp::Vec r(r_);
  auto c = org_sp::get_c(w, h, c_);
  for(int y = 0; y < h; y++) { // Loop over image rows
    org_sp::compute_inner(y, w, h, samps, cam, cx, cy, r, c.data());
  }
  org_sp::assign_c(w, h, c, c_);
}
void compute_org_sp_openmp(
  void*, int w, int h, int samps, Ray cam_, Vec cx_, Vec cy_, Vec r_, Vec *c_) {
  org_sp::Ray cam(cam_);
  org_sp::Vec cx(cx_);
  org_sp::Vec cy(cy_);
  org_sp::Vec r(r_);
  auto c = org_sp::get_c(w, h, c_);
  #pragma omp parallel for schedule(dynamic, 1) private(r)
  for(int y = 0; y < h; y++) { // Loop over image rows
    org_sp::compute_inner(y, w, h, samps, cam, cx, cy, r, c.data());
  }
  org_sp::assign_c(w, h, c, c_);
}
