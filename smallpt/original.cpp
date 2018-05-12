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

using Vec = Vec_detail<double>;
using Ray = Ray_detail<double>;
using Sphere = Sphere_detail<double>;

namespace org {
Sphere spheres[] = {
    // Scene: radius, position, emission, color, material
    Sphere(1e5, Vec(1e5 + 1, 40.8, 81.6), Vec(), Vec(.75, .25, .25),
           DIFF),  // Left
    Sphere(1e5, Vec(-1e5 + 99, 40.8, 81.6), Vec(), Vec(.25, .25, .75),
           DIFF),                                                      // Rght
    Sphere(1e5, Vec(50, 40.8, 1e5), Vec(), Vec(.75, .75, .75), DIFF),  // Back
    Sphere(1e5, Vec(50, 40.8, -1e5 + 170), Vec(), Vec(), DIFF),        // Frnt
    Sphere(1e5, Vec(50, 1e5, 81.6), Vec(), Vec(.75, .75, .75), DIFF),  // Botm
    Sphere(1e5, Vec(50, -1e5 + 81.6, 81.6), Vec(), Vec(.75, .75, .75),
           DIFF),                                                       // Top
    Sphere(16.5, Vec(27, 16.5, 47), Vec(), Vec(1, 1, 1) * .999, SPEC),  // Mirr
    Sphere(16.5, Vec(73, 16.5, 78), Vec(), Vec(1, 1, 1) * .999, REFR),  // Glas
    Sphere(600, Vec(50, 681.6 - .27, 81.6), Vec(12, 12, 12), Vec(),
           DIFF)  // Lite
};
inline double clamp(double x) {
  return x < 0 ? 0 : x > 1 ? 1 : x;
}
inline bool intersect(const Ray& r, double& t, int& id) {
  double n = sizeof(spheres) / sizeof(Sphere), d, inf = t = 1e20;
  for (int i = int(n); i--;) {
    if ((d = spheres[i].intersect(r)) && d < t) {
      t = d;
      id = i;
    }
  }
  return t < inf;
}
Vec radiance(const Ray& r, int depth, uint16_t* Xi) {
  double t;    // distance to intersection
  int id = 0;  // id of intersected object
  if (!intersect(r, t, id)) {
    return Vec();  // if miss, return black
  }
  const Sphere& obj = spheres[id];  // the hit object
  Vec x = r.o + r.d * t;
  Vec n = (x - obj.p).norm(), nl = n.dot(r.d) < 0 ? n : n * -1, f = obj.c;
  double p = f.x > f.y && f.x > f.z ? f.x : f.y > f.z ? f.y : f.z;  // max refl
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
    double r1 = 2 * M_PI * get_random(Xi), r2 = get_random(Xi), r2s = sqrt(r2);
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
  double nc = 1, nt = 1.5, nnt = into ? nc / nt : nt / nc, ddn = r.d.dot(nl),
         cos2t;
  if ((cos2t = 1 - nnt * nnt * (1 - ddn * ddn)) <
      0) {  // Total internal reflection
    return obj.e + f.mult(radiance(reflRay, depth, Xi));
  }
  Vec tdir =
      (r.d * nnt - n * ((into ? 1 : -1) * (ddn * nnt + sqrt(cos2t)))).norm();
  double a = nt - nc;
  double b = nt + nc, R0 = a * a / (b * b), c = 1 - (into ? -ddn : tdir.dot(n));
  double Re = R0 + (1 - R0) * c * c * c * c * c;
  double Tr = 1 - Re, P = .25 + .5 * Re, RP = Re / P, TP = Tr / (1 - P);
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
          double r1 = 2 * get_random(Xi),
                 dx = r1 < 1 ? sqrt(r1) - 1 : 1 - sqrt(2 - r1);
          double r2 = 2 * get_random(Xi),
                 dy = r2 < 1 ? sqrt(r2) - 1 : 1 - sqrt(2 - r2);
          Vec d = cx * (((sx + .5 + dx) / 2 + x) / w - .5) +
                  cy * (((sy + .5 + dy) / 2 + y) / h - .5) + cam.d;
          r = r +
              radiance(Ray(cam.o + d * 140, d.norm()), 0, Xi) * (1. / samps);
        }  // Camera rays are pushed ^^^^^ forward to start in interior
        c[i] = c[i] + Vec(clamp(r.x), clamp(r.y), clamp(r.z)) * .25;
      }
    }
  }
}
}  // namespace org
void compute_org(void*, int w, int h, int samps, Ray cam, Vec cx, Vec cy, Vec r,
                 Vec* c) {
  for (int y = 0; y < h; y++) {  // Loop over image rows
    org::compute_inner(y, w, h, samps, cam, cx, cy, r, c);
  }
}
void compute_org_openmp(void*, int w, int h, int samps, Ray cam, Vec cx, Vec cy,
                        Vec r, Vec* c) {
#pragma omp parallel for schedule(dynamic, 1) private(r)
  for (int y = 0; y < h; y++) {  // NOLINT
    // Loop over image rows
    org::compute_inner(y, w, h, samps, cam, cx, cy, r, c);
  }
}
