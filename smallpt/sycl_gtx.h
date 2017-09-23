#pragma once

// smallpt, a Path Tracer by Kevin Beason, 2008
//
// Modified by Peter Žužek
// For the original code, see github.com/munificient/smallpt
// For the original license, see smallpt.LICENSE.txt

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <iostream>

#define SYCL_SIMPLE_SWIZZLES
#include <CL/sycl.hpp>

#include "classes.h"
#include "win.h"

#ifndef float_type
#define float_type double
#endif
#ifndef modify_sample_rate
#define modify_sample_rate 1
#endif

using Vec = Vec_detail<float_type>;
using Ray = Ray_detail<float_type>;
using Sphere = Sphere_detail<float_type, modify_sample_rate>;

#ifndef SYCL_GTX
#include <CL/sycl_gtx_compatibility.h>
#endif

namespace ns_sycl_gtx {

using namespace cl::sycl;

static const int numSpheres = 9;
static Sphere spheres[numSpheres] = {
    // Scene: radius, position, emission, color, material
    Sphere(1e4, Vec(1e4 + 1, 40.8, 81.6), Vec(), Vec(.75, .25, .25),
           DIFF),  // Left
    Sphere(1e4, Vec(-1e4 + 99, 40.8, 81.6), Vec(), Vec(.25, .25, .75),
           DIFF),                                                      // Rght
    Sphere(1e4, Vec(50, 40.8, 1e4), Vec(), Vec(.75, .75, .75), DIFF),  // Back
    Sphere(1e4, Vec(50, 40.8, -1e4 + 170), Vec(), Vec(), DIFF),        // Frnt
    Sphere(1e4, Vec(50, 1e4, 81.6), Vec(), Vec(.75, .75, .75), DIFF),  // Botm
    Sphere(1e4, Vec(50, -1e4 + 81.6, 81.6), Vec(), Vec(.75, .75, .75),
           DIFF),                                                       // Top
    Sphere(16.5, Vec(27, 16.5, 47), Vec(), Vec(1, 1, 1) * .999, SPEC),  // Mirr
    Sphere(16.5, Vec(73, 16.5, 78), Vec(), Vec(1, 1, 1) * .999, REFR),  // Glas
    Sphere(600, Vec(50, 681.6 - .27, 81.6), Vec(12, 12, 12), Vec(),
           DIFF)  // Lite
};

using spheres_t =
    accessor<float16, 1, access::mode::read, access::target::global_buffer>;

struct Vector : public ::Vec_detail<float1> {
 private:
  using Base = ::Vec_detail<float1>;

 public:
  Vector(float x = 0, float y = 0, float z = 0) : Base(x, y, z) {}
  Vector(const ::Vec_detail<float_type>& base)
      : Base(static_cast<float>(base.x), static_cast<float>(base.y),
             static_cast<float>(base.z)) {}
  template <typename t = float1>
  Vector(const Base& base,
         typename std::enable_if<!std::is_same<t, float_type>::value>::type* =
             nullptr)
      : Base(base) {}
  Vector(float3 data) : Base(data.x(), data.y(), data.z()) {}
};

using RaySycl = ::Ray_detail<float1>;

struct SphereSycl : public ::Sphere_detail<float1> {
  float1 refl;

  SphereSycl(const float16& data)
      : ::Sphere_detail<float1>(
            data.lo().lo().w(), Vector(data.lo().lo().xyz()),
            Vector(data.lo().hi().xyz()), Vector(data.hi().lo().xyz()),
            Refl_t::DIFF  // Not important
            ),
        refl(data.hi().lo().w()) {}

  float1 intersect(
      const Ray_detail<float1>& r) const {  // returns distance, 0 if no hit
    float1 return_vec;
    Vector op = p - r.o;  // Solve t^2*d.d + 2*t*(o-p).d + (o-p).(o-p)-R^2 = 0
    float1 t;
    float1 eps = 1e-2f;
    float1 b = op.dot(r.d);
    float1 det = b * b - op.dot(op) + rad * rad;

    SYCL_IF(det < 0) {
      return_vec = 0;
    }
    SYCL_ELSE {
      det = cl::sycl::sqrt(det);
      t = b - det;
      SYCL_IF(t > eps) {
        return_vec = t;
      }
      SYCL_ELSE {
        t = b + det;
        SYCL_IF(t > eps) {
          return_vec = t;
        }
        SYCL_ELSE {
          return_vec = 0;
        }
        SYCL_END;
      }
      SYCL_END;
    }
    SYCL_END;

    return return_vec;
  }
};

inline void clamp(float1& x) {
  SYCL_IF(x < 0) {
    x = 0;
  }
  SYCL_ELSE_IF(x > 1) {
    x = 1;
  }
  SYCL_END;
}

inline bool1 intersect(spheres_t spheres, const RaySycl& r, float1& t,
                       int1& id) {
  using namespace cl::sycl;
  float1 d;
  float1 inf = t = 1e20f;

  int1 i = ns_sycl_gtx::numSpheres;
  SYCL_WHILE(i > 0) {
    i -= 1;
    d = SphereSycl(spheres[i]).intersect(r);
    SYCL_IF(d != 0 && d < t) {
      t = d;
      id = i;
    }
    SYCL_END;
  }
  SYCL_END;

  return t < inf;
}

// http://stackoverflow.com/a/16077942
static float1 getRandom(uint2& seed) {
  // Note: Should not be declared static
  const float1 invMaxInt = 1.0f / 4294967296.0f;
  uint1 x = seed.x() * 17 + seed.y() * 13123;
  seed.x() = (x << 13) ^ x;
  seed.y() = seed.y() ^ x << 7;
  return static_cast<float1>((x * (x * x * 15731 + 74323) + 871483) *
                             invMaxInt);
}

static void radiance(Vector& return_vec, spheres_t spheres, RaySycl r,
                     uint2& randomSeed, Vector cl = {0, 0, 0},
                     Vector cf = {1, 1, 1}) {
  using namespace cl::sycl;

  float1 t;     // distance to intersection
  int1 id = 0;  // id of intersected object
  int1 depth = 0;

  // cl is accumulated color
  // cf is accumulated reflectance

  RaySycl reflRay(Vector(0), Vector(0));
  Vector x, tdir;

  SYCL_WHILE(true) {
    SYCL_IF(!intersect(spheres, r, t, id)) {
      // if miss, don't add anything
      return_vec = cl;
      SYCL_BREAK;
    }
    SYCL_END;

    auto obj = SphereSycl(spheres[id]);  // the hit object
    x = r.o + r.d * t;

    Vector n = Vector(x - obj.p).norm();
    Vector nl = n;
    SYCL_IF(n.dot(r.d) > 0) {
      nl = nl * -1;
    }
    SYCL_END;

    Vector f = obj.c;

    float1 p;  // max refl
    SYCL_IF(f.x > f.y && f.x > f.z) {
      p = f.x;
    }
    SYCL_ELSE_IF(f.y > f.z) {
      p = f.y;
    }
    SYCL_ELSE {
      p = f.z;
    }
    SYCL_END;

    cl = cl + cf.mult(obj.e);

    depth += 1;
    SYCL_IF(depth > 5) {
      SYCL_IF(getRandom(randomSeed) < p) {
        f = f * (1 / p);
      }
      SYCL_ELSE {
        return_vec = cl;
        SYCL_BREAK;
      }
      SYCL_END;
    }
    SYCL_END;

    cf = cf.mult(f);

    SYCL_IF(obj.refl == (::cl_float)DIFF) {  // Ideal DIFFUSE reflection
      float1 r1 = static_cast<float1>(2 * M_PI * getRandom(randomSeed));
      float1 r2 = getRandom(randomSeed);
      float1 r2s = cl::sycl::sqrt(r2);
      Vector w = nl;

      Vector u(0, 0, 0);
      SYCL_IF(cl::sycl::fabs(w.x) > .1f) {
        u.y = 1;
      }
      SYCL_ELSE {
        u.x = 1;
      }
      SYCL_END;
      u = (u % w).norm();

      Vector v = w % u;
      Vector d =
          Vector(u * cl::sycl::cos(r1) * r2s + v * cl::sycl::sin(r1) * r2s +
                 w * cl::sycl::sqrt(1 - r2))
              .norm();

      // Recursion
      r = RaySycl(x, d);
      SYCL_CONTINUE;
    }
    SYCL_ELSE_IF(obj.refl == (::cl_float)SPEC) {  // Ideal SPECULAR reflection
      // Recursion
      r = RaySycl(x, r.d - n * 2 * n.dot(r.d));
      SYCL_CONTINUE;
    }
    SYCL_END;

    reflRay =
        RaySycl(x, r.d - n * 2 * n.dot(r.d));  // Ideal dielectric REFRACTION
    bool1 into = n.dot(nl) > 0;                // Ray from outside going in?
    float1 nc = 1;
    float1 nt = 1.5f;

    float1 nnt;
    SYCL_IF(into) {
      nnt = nc / nt;
    }
    SYCL_ELSE {
      nnt = nt / nc;
    }
    SYCL_END;

    float1 ddn = r.d.dot(nl);
    float1 cos2t = 1 - nnt * nnt * (1 - ddn * ddn);
    SYCL_IF(cos2t < 0) {  // Total internal reflection
      // Recursion
      r = reflRay;
      SYCL_CONTINUE;
    }
    SYCL_END;

    float1 tmp = 1;
    SYCL_IF(!into) {
      tmp = -1;
    }
    SYCL_END;

    tdir = Vector(r.d * nnt - n * (tmp * (ddn * nnt + cl::sycl::sqrt(cos2t))))
               .norm();
    float1 a = nt - nc;
    float1 b = nt + nc;
    float1 R0 = a * a / (b * b);

    float1 c = 1;
    SYCL_IF(into) {
      c += ddn;
    }
    SYCL_ELSE {
      c -= tdir.dot(n);
    }
    SYCL_END;

    float1 Re = R0 + (1 - R0) * c * c * c * c * c;
    float1 Tr = 1 - Re;
    float1 P = .25f + .5f * Re;
    float1 RP = Re / P;
    float1 TP = Tr / (1 - P);

    // Tail recursion
    SYCL_IF(getRandom(randomSeed) < P) {
      cf = cf * RP;
      r = reflRay;
    }
    SYCL_ELSE {
      cf = cf * TP;
      r = RaySycl(x, tdir);
    }
    SYCL_END;
  }
  SYCL_END;
}

}  // namespace ns_sycl_gtx

static void compute_sycl_gtx(void* dev, int w, int h, int samps, Ray cameraRay,
                             Vec cxIn, Vec cyIn, Vec rIn, Vec* cVecOut) {
  using namespace std;
  using namespace cl::sycl;
  using namespace ns_sycl_gtx;

  queue q(*static_cast<device*>(dev));  // NOLINT

  auto spheres_tmp = buffer<float16>(range<1>(ns_sycl_gtx::numSpheres));
  {
// TODO(progtx): Conform to the SYCL spec
#ifdef SYCL_GTX
    auto assign = [](cl::sycl::cl_float4& target, ::Vec& data) {
#else
    auto assign = [](cl::sycl::cl_float4 target, ::Vec& data) {
#endif
      using type = float4::element_type;
      target.x() = static_cast<type>(data.x);
      target.y() = static_cast<type>(data.y);
      target.z() = static_cast<type>(data.z);
    };

    auto s = spheres_tmp.get_access<access::mode::discard_write,
                                    access::target::host_buffer>();
    // See SphereSycl
    for (int i = 0; i < ns_sycl_gtx::numSpheres; ++i) {
      auto& si = s[i];
      auto& sj = ns_sycl_gtx::spheres[i];

      assign(si.lo().lo(), sj.p);
      assign(si.lo().hi(), sj.e);
      assign(si.hi().lo(), sj.c);

      si.lo().lo().w() = static_cast<::cl_float>(sj.rad);
      si.hi().lo().w() = static_cast<::cl_float>(sj.refl);
    }
  }

  // Divide GPU calculation into parts to prevent monitor freezing
  const int numParts = (q.get_device().get_info<info::device::device_type>() ==
                        info::device_type::gpu)
                           ? std::min(samps / 5 + 1, h / 10)
                           : 1;

  uint16_t Xi[3] = {0, 0, 0};
  vector<buffer<float3>> colors;
  vector<buffer<cl::sycl::cl_uint2>> seeds_tmp;
  vector<pair<int, int>> lineOffset;
  auto starts_tmp = buffer<int>(range<1>(numParts));
  {
    auto starts = starts_tmp.get_access<access::mode::discard_write,
                                        access::target::host_buffer>();

    for (int k = 0; k < numParts; ++k) {
      int start = h * k / numParts;
      int end = h * (k + 1) / numParts;
      int height = end - start;
      range<1> length(height * w);

      lineOffset.emplace_back(start, height);
      colors.emplace_back(length);
      seeds_tmp.emplace_back(length);
      starts[k] = start;

      auto seeds = seeds_tmp.back()
                       .get_access<access::mode::discard_write,
                                   access::target::host_buffer>();

      for (int y = 0; y < height; ++y) {
        Xi[2] = y * y * y;
        for (int x = 0; x < w; ++x) {
          cl::sycl::cl_uint2 seed;
          seed.x() = static_cast<::cl_uint>(get_random(Xi));
          seed.y() = static_cast<::cl_uint>(get_random(Xi));
          seeds[y * w + x] = seed;
        }
      }
    }
  }

  for (auto k = 0; k < numParts; ++k) {
    q.submit([&](handler& cgh) {
      auto c = colors[k]
                   .get_access<access::mode::discard_read_write,
                               access::target::global_buffer>(cgh);

      // TODO(progtx): constant_buffer
      auto spheres = spheres_tmp.get_access<access::mode::read,
                                            access::target::global_buffer>(cgh);
      auto seeds =
          seeds_tmp[k]
              .get_access<access::mode::read, access::target::global_buffer>(
                  cgh);
      auto starts = starts_tmp.get_access<access::mode::read,
                                          access::target::global_buffer>(cgh);

      cgh.parallel_for<class smallpt>(
          range<2>(w, lineOffset[k].second), [=](id<2> i) {
            Vector cx(cxIn);
            Vector cy(cyIn);
            Vector r(rIn);
            RaySycl cam(Vector(cameraRay.o), Vector(cameraRay.d));
            uint2 randomSeed;
            randomSeed.x() = seeds[i].x() * i[0] + i[0] + 1;
            randomSeed.y() = seeds[i].y() * i[1] + i[1] + 1;

            c[i] = 0;  // Important to start at zero

            // 2x2 subpixel rows
            SYCL_FOR(int1 sy = 0, sy < 2, sy++) {
              // 2x2 subpixel cols
              SYCL_FOR(int1 sx = 0, sx < 2, sx++) {
                SYCL_FOR(int1 s = 0, s < samps, s++) {
                  float2 rnew;
                  rnew.x() = 2 * getRandom(randomSeed);
                  rnew.y() = 2 * getRandom(randomSeed);

                  float2 dd;

                  SYCL_IF(rnew.x() < 1) {
                    dd.x() = cl::sycl::sqrt(rnew.x()) - 1;
                  }
                  SYCL_ELSE {
                    dd.x() = 1 - cl::sycl::sqrt(2 - rnew.x());
                  }
                  SYCL_END;

                  SYCL_IF(rnew.y() < 1) {
                    dd.y() = cl::sycl::sqrt(rnew.y()) - 1;
                  }
                  SYCL_ELSE {
                    dd.y() = 1 - cl::sycl::sqrt(2 - rnew.y());
                  }
                  SYCL_END;

                  Vector d =
                      (cx * (((sx + .5f + dd.x()) / 2 + i[0]) / w - .5f)) +
                      (cy * (((sy + .5f + dd.y()) / 2 + i[1] + starts[k]) / h -
                             .5f)) +
                      cam.d;

                  // TODO(progtx):
                  Vector rad;
                  radiance(rad, spheres, RaySycl(cam.o + d * 140, d.norm()),
                           randomSeed);
                  r = r + rad * (1.f / samps);
                }  // Camera rays are pushed ^^^^^ forward to start in interior
                SYCL_END;

                ns_sycl_gtx::clamp(r.x);
                ns_sycl_gtx::clamp(r.y);
                ns_sycl_gtx::clamp(r.z);

                r = r * .25f;
                auto ci = c[i];
                ci.x() = ci.x() + r.x;
                ci.y() = ci.y() + r.y;
                ci.z() = ci.z() + r.z;

                r = Vector();
              }
              SYCL_END;
            }
            SYCL_END;
          });
    });
  }

  auto assign = [](::Vec& target, cl::sycl::cl_float3& data) {
    target.x = static_cast<float_type>(data.x());
    target.y = static_cast<float_type>(data.y());
    target.z = static_cast<float_type>(data.z());
  };

  for (auto k = 0; k < numParts; ++k) {
#ifndef NDEBUG
    cout << "Waiting for kernel to finish ..." << endl;
#endif
    auto c =
        colors[k].get_access<access::mode::read, access::target::host_buffer>();

#ifndef NDEBUG
    cout << "Copying results ..." << endl;
#endif
    auto start = lineOffset[k].first;
    auto end = start + lineOffset[k].second;
    for (int y = start; y < end; ++y) {
      int i = (y - start) * w;
      int ri = (h - y - 1) * w;  // Picture is upside down
      for (int x = 0; x < w; ++x) {
        auto& ci = c[i];
        assign(cVecOut[ri], ci);

        ++i;
        ++ri;
      }
    }
  }
}
