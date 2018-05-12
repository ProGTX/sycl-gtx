// smallpt, a Path Tracer by Kevin Beason, 2008
//
// Modified by Peter Žužek
// For the original code, see github.com/munificient/smallpt
// For the original license, see smallpt.LICENSE.txt

#include <math.h>    // smallpt, a Path Tracer by Kevin Beason, 2008
#include <stdio.h>   //        Remove "-fopenmp" for g++ version < 4.2
#include <stdlib.h>  // Make : g++ -O3 -fopenmp smallpt.cpp -o smallpt
// Usage: time ./smallpt 5000 && xv image.ppm
// position, also color (r,g,b)

#include "classes.h"
#include "win.h"
#include <vector>

using Vec = Vec_detail<double>;
using Ray = Ray_detail<double>;
using Sphere = Sphere_detail<double>;

namespace org_sp {
struct Vec {
  float x, y, z;
  Vec(float x = 0, float y = 0, float z = 0) {
    this->x = x;
    this->y = y;
    this->z = z;
  }
  Vec(const ::Vec& v)
      : Vec(static_cast<float>(v.x), static_cast<float>(v.y),
            static_cast<float>(v.z)) {}
  Vec operator+(const Vec& b) const {
    return Vec(x + b.x, y + b.y, z + b.z);
  }
  Vec operator-(const Vec& b) const {
    return Vec(x - b.x, y - b.y, z - b.z);
  }
  Vec operator*(float b) const {
    return Vec(x * b, y * b, z * b);
  }
  Vec mult(const Vec& b) const {
    return Vec(x * b.x, y * b.y, z * b.z);
  }
  Vec& norm() {
    return *this = *this * (1 / sqrt(x * x + y * y + z * z));
  }
  float dot(const Vec& b) const {
    return this->x * b.x + this->y * b.y + this->z * b.z;
  }  // cross:
  Vec operator%(Vec& b) {
    return Vec(this->y * b.z - this->z * b.y, this->z * b.x - this->x * b.z,
               this->x * b.y - this->y * b.x);
  }
};
struct Ray {
  Vec o, d;
  Ray(Vec o, Vec d) : o(o), d(d) {}
  Ray(const ::Ray& r) : o(r.o), d(r.d) {}
};
enum Refl_t { DIFF, SPEC, REFR };  // material types, used in radiance()
struct Sphere {
  float rad;    // radius
  Vec p, e, c;  // position, emission, color
  Refl_t refl;  // reflection type (DIFFuse, SPECular, REFRactive)
  Sphere(float rad, Vec p, Vec e, Vec c, Refl_t refl)
      : rad(rad), p(p), e(e), c(c), refl(refl) {}
  float intersect(const Ray& r) const {  // returns distance, 0 if nohit
    Vec op = p - r.o;  // Solve t^2*d.d + 2*t*(o-p).d + (o-p).(o-p)-R^2 = 0
    float t, eps = 1e-2f, b = op.dot(r.d), det = b * b - op.dot(op) + rad * rad;
    if (det < 0) {
      return 0;
    } else {
      det = sqrt(det);
    }
    return (t = b - det) > eps ? t : ((t = b + det) > eps ? t : 0);
  }
};
Sphere spheres[] = {
    // Scene: radius, position, emission, color, material
    Sphere(1e4f, Vec(1e4f + 1, 40.8f, 81.6f), Vec(), Vec(.75f, .25f, .25f),
           DIFF),  // Left
    Sphere(1e4f, Vec(-1e4f + 99, 40.8f, 81.6f), Vec(), Vec(.25f, .25f, .75f),
           DIFF),  // Rg
    Sphere(1e4f, Vec(50, 40.8f, 1e4f), Vec(), Vec(.75f, .75f, .75f),
           DIFF),                                                   // Back
    Sphere(1e4f, Vec(50, 40.8f, -1e4f + 170), Vec(), Vec(), DIFF),  // Front
    Sphere(1e4f, Vec(50, 1e4f, 81.6f), Vec(), Vec(.75f, .75f, .75f),
           DIFF),  // Bottom
    Sphere(1e4f, Vec(50, -1e4f + 81.6f, 81.6f), Vec(), Vec(.75f, .75f, .75f),
           DIFF),  // Tp
    Sphere(16.5f, Vec(27, 16.5f, 47), Vec(), Vec(1, 1, 1) * .999f,
           SPEC),  // Mirr
    Sphere(16.5f, Vec(73, 16.5f, 78), Vec(), Vec(1, 1, 1) * .999f,
           REFR),  // Glas
    Sphere(600, Vec(50, 681.6f - .27f, 81.6f), Vec(12, 12, 12), Vec(),
           DIFF)  // Lite
};
inline float clamp(float x) {
  return x < 0 ? 0 : x > 1 ? 1 : x;
}
inline bool intersect(const Ray& r, float& t, int& id) {
  float n = sizeof(spheres) / sizeof(Sphere), d, inf = t = 1e20f;
  for (int i = int(n); i--;) {
    if ((d = spheres[i].intersect(r)) && d < t) {
      t = d;
      id = i;
    }
  }
  return t < inf;
}
Vec radiance(const Ray& r, int depth, uint16_t* Xi) {
  float t;     // distance to intersection
  int id = 0;  // id of intersected object
  if (!intersect(r, t, id)) {
    return Vec();  // if miss, return black
  }
  const Sphere& obj = spheres[id];  // the hit object
  Vec x = r.o + r.d * t;
  Vec n = (x - obj.p).norm(), nl = n.dot(r.d) < 0 ? n : n * -1, f = obj.c;
  float p = f.x > f.y && f.x > f.z ? f.x : f.y > f.z ? f.y : f.z;  // max refl
  if (depth > 255) {
    return obj.e;
  }
  if (++depth > 5) {
    if (get_random(Xi) < p) {
      f = f * (1 / p);
    } else {
      return obj.e;
    }
  }                        // R.R.
  if (obj.refl == DIFF) {  // Ideal DIFFUSE reflection
    float r1 = static_cast<float>(2 * M_PI * get_random(Xi));
    float r2 = static_cast<float>(get_random(Xi)), r2s = sqrt(r2);
    Vec w = nl, u = ((fabs(w.x) > .1 ? Vec(0, 1) : Vec(1)) % w).norm(),
        v = w % u;
    Vec d = (u * cos(r1) * r2s + v * sin(r1) * r2s + w * sqrt(1 - r2)).norm();
    return obj.e + f.mult(radiance(Ray(x, d), depth, Xi));
  } else if (obj.refl == SPEC) {  // Ideal SPECULAR reflection
    return obj.e +
           f.mult(radiance(Ray(x, r.d - n * 2 * n.dot(r.d)), depth, Xi));
  }
  Ray reflRay(x, r.d - n * 2 * n.dot(r.d));  // Ideal dielectric REFRACTION
  bool into = n.dot(nl) > 0;                 // Ray from outside going in?
  float nc = 1, nt = 1.5f, nnt = into ? nc / nt : nt / nc, ddn = r.d.dot(nl),
        cos2t;
  if ((cos2t = 1 - nnt * nnt * (1 - ddn * ddn)) <
      0) {  // Total internal reflection
    return obj.e + f.mult(radiance(reflRay, depth, Xi));
  }
  Vec tdir =
      (r.d * nnt - n * ((into ? 1 : -1) * (ddn * nnt + sqrt(cos2t)))).norm();
  float a = nt - nc;
  float b = nt + nc, R0 = a * a / (b * b), c = 1 - (into ? -ddn : tdir.dot(n));
  float Re = R0 + (1 - R0) * c * c * c * c * c;
  float Tr = 1 - Re, P = .25f + .5f * Re, RP = Re / P, TP = Tr / (1 - P);
  return obj.e + f.mult(depth > 2
                            ? (get_random(Xi) < P
                                   ?  // Russian roulette
                                   radiance(reflRay, depth, Xi) * RP
                                   : radiance(Ray(x, tdir), depth, Xi) * TP)
                            : radiance(reflRay, depth, Xi) * Re +
                                  radiance(Ray(x, tdir), depth, Xi) * Tr);
}
inline void compute_inner(int y, int w, int h, int samps, Ray& cam, Vec& cx,
                          Vec& cy, Vec& r, Vec* c) {
  // fprintf(stderr, "\rRendering (%d spp) %5.2f%%", samps * 4, 100.*y / (h -
  // 1));
  // Loop cols
  for (uint16_t x = 0, Xi[3] = {0, 0, static_cast<uint16_t>(y * y * y)}; x < w;
       x++) {
    for (int sy = 0, i = (h - y - 1) * w + x; sy < 2;
         sy++) {                                   // 2x2 subpixel rows
      for (int sx = 0; sx < 2; sx++, r = Vec()) {  // 2x2 subpixel cols
        for (int s = 0; s < samps; s++) {
          float r1 = static_cast<float>(2 * get_random(Xi)),
                dx = r1 < 1 ? sqrt(r1) - 1 : 1 - sqrt(2 - r1);
          float r2 = static_cast<float>(2 * get_random(Xi)),
                dy = r2 < 1 ? sqrt(r2) - 1 : 1 - sqrt(2 - r2);
          Vec d = cx * (((sx + .5f + dx) / 2 + x) / w - .5f) +
                  cy * (((sy + .5f + dy) / 2 + y) / h - .5f) + cam.d;
          r = r +
              radiance(Ray(cam.o + d * 140, d.norm()), 0, Xi) * (1.f / samps);
        }  // Camera rays are pushed ^^^^^ forward to start in interior
        c[i] = c[i] + Vec(clamp(r.x), clamp(r.y), clamp(r.z)) * .25f;
      }
    }
  }
}
std::vector<org_sp::Vec> get_c(int w, int h, ::Vec* cVecOut) {
  std::vector<org_sp::Vec> c;
  c.reserve(static_cast<size_t>(w) * static_cast<size_t>(h));
  for (int y = 0; y < h; ++y) {
    for (int x = 0; x < w; ++x) {
      int i = y * w + x;
      c.emplace_back(cVecOut[i]);
    }
  }
  return c;
}
void assign_c(int w, int h, std::vector<org_sp::Vec>& c, ::Vec* cVecOut) {
  for (int y = 0; y < h; ++y) {
    for (int x = 0; x < w; ++x) {
      int i = y * w + x;
      cVecOut[i].x = c[i].x;
      cVecOut[i].y = c[i].y;
      cVecOut[i].z = c[i].z;
    }
  }
}
}  // namespace org_sp
void compute_org_sp(void*, int w, int h, int samps, Ray cameraRay, Vec cxIn,
                    Vec cyIn, Vec rIn, Vec* cVecOut) {
  org_sp::Ray cam(cameraRay);
  org_sp::Vec cx(cxIn);
  org_sp::Vec cy(cyIn);
  org_sp::Vec r(rIn);
  auto c = org_sp::get_c(w, h, cVecOut);
  for (int y = 0; y < h; y++) {  // Loop over image rows
    org_sp::compute_inner(y, w, h, samps, cam, cx, cy, r, c.data());
  }
  org_sp::assign_c(w, h, c, cVecOut);
}
void compute_org_sp_openmp(void*, int w, int h, int samps, Ray cameraRay,
                           Vec cxIn, Vec cyIn, Vec rIn, Vec* cVecOut) {
  org_sp::Ray cam(cameraRay);
  org_sp::Vec cx(cxIn);
  org_sp::Vec cy(cyIn);
  org_sp::Vec r(rIn);
  auto c = org_sp::get_c(w, h, cVecOut);
#pragma omp parallel for schedule(dynamic, 1) private(r)
  for (int y = 0; y < h; y++) {  // NOLINT
    // Loop over image rows
    org_sp::compute_inner(y, w, h, samps, cam, cx, cy, r, c.data());
  }
  org_sp::assign_c(w, h, c, cVecOut);
}
