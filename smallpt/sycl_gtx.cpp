// smallpt, a Path Tracer by Kevin Beason, 2008
//
// Modified by Peter Žužek
// For the original code, see github.com/munificient/smallpt
// For the original license, see smallpt.LICENSE.txt

#include <cmath>
#include <cstdlib>
#include <cstdio>
#include <iostream>

#include <sycl.hpp>

#include "classes.h"
#include "msvc.h"

namespace ns_sycl_gtx {

using namespace cl::sycl;

static const int numSpheres = 9;
Sphere spheres[numSpheres] = {//Scene: radius, position, emission, color, material
	Sphere(1e4, Vec(1e4 + 1, 40.8, 81.6), Vec(), Vec(.75, .25, .25), DIFF),//Left
	Sphere(1e4, Vec(-1e4 + 99, 40.8, 81.6), Vec(), Vec(.25, .25, .75), DIFF),//Rght
	Sphere(1e4, Vec(50, 40.8, 1e4), Vec(), Vec(.75, .75, .75), DIFF),//Back
	Sphere(1e4, Vec(50, 40.8, -1e4 + 170), Vec(), Vec(), DIFF),//Frnt
	Sphere(1e4, Vec(50, 1e4, 81.6), Vec(), Vec(.75, .75, .75), DIFF),//Botm
	Sphere(1e4, Vec(50, -1e4 + 81.6, 81.6), Vec(), Vec(.75, .75, .75), DIFF),//Top
	Sphere(16.5, Vec(27, 16.5, 47), Vec(), Vec(1, 1, 1)*.999, SPEC),//Mirr
	Sphere(16.5, Vec(73, 16.5, 78), Vec(), Vec(1, 1, 1)*.999, REFR),//Glas
	Sphere(600, Vec(50, 681.6 - .27, 81.6), Vec(12, 12, 12), Vec(), DIFF) //Lite
};

using spheres_t = accessor<float16, 1, access::read, access::global_buffer>;

#ifdef SYCL_GTX
struct Vector : public float3 {
	Vector()
		: float3(0, 0, 0) {}
	Vector(const Vec& v)
		: float3((float)v.x, (float)v.y, (float)v.z) {}
	template <class T>
	Vector(T n)
		: float3(n) {}
	template <class X, class Y, class Z>
	Vector(X&& x, Y&& y, Z&& z)
		: float3(x, y, z) {}

	template <class T>
	Vector& operator=(T&& t) {
		float3::operator=(t);
		return *this;
	}

	Vector mult(const Vector &b) const {
		return Vector(x*b.x, y*b.y, z*b.z);
	}
	Vector& norm() {
		operator*=(1 / sqrt(x*x + y*y + z*z));
		return *this;
	}
	float1 dot(const Vector &b) const {
		return x*b.x + y*b.y + z*b.z;
	}
	Vector operator%(Vector&b) {
		return Vector(y*b.z - z*b.y, z*b.x - x*b.z, x*b.y - y*b.x);
	}
};

#else

using Vector = ::Vec;

#endif // SYCL_GTX

struct RaySycl {
	Vector o, d;
	RaySycl(const ::Ray& r)
		: o(r.o), d(r.d) {}
	RaySycl(Vector o_, Vector d_)
		: o(o_), d(d_) {}
	RaySycl& operator=(const RaySycl&) = default;
};

struct SphereSycl {
private:
	float16 data;

public:
	template <class T>
	SphereSycl(T&& data)
		: data(std::forward<T>(data)) {}

	float1& rad() const {
		return data.w;
	}
	float3& p() const { // position
		return data.xyz;
	}
	float3& e() { // emission
		return data.lo.hi.xyz;
	}
	float3& c() const { // color
		return data.hi.xyz;
	}
	float1 refl() const { // reflection type (Refl_t)
		return data.hi.w;
	}

	void intersect(float1& return_, const RaySycl& r) const { // returns distance, 0 if no hit
		Vector op = p() - r.o; // Solve t^2*d.d + 2*t*(o-p).d + (o-p).(o-p)-R^2 = 0
		float1 t;
		float1 eps = 1e-2f;
		float1 b = op.dot(r.d);
		float1 det = b*b - op.dot(op) + rad()*rad();

		SYCL_IF(det < 0)
			return_ = 0;
		SYCL_ELSE {
			det = sqrt(det);
			t = b - det;
			SYCL_IF(t > eps)
				return_ = t;
			SYCL_ELSE {
				t = b + det;
				SYCL_IF(t > eps)
					return_ = t;
				SYCL_ELSE
					return_ = 0;
				SYCL_END
			}
			SYCL_END
		}
		SYCL_END
	}
};

inline void clamp(float1& x) {
	SYCL_IF(x < 0)
		x = 0;
	SYCL_ELSE_IF(x > 1)
		x = 1;
	SYCL_END
}

inline bool1 intersect(spheres_t spheres, const RaySycl& r, float1& t, int1& id) {
	using namespace cl::sycl;
	float1 d;
	float1 inf = t = 1e20f;

	int1 i = ns_sycl_gtx::numSpheres;
	SYCL_WHILE(i > 0) {
		i -= 1;
		SphereSycl(spheres[i]).intersect(d, r);
		SYCL_IF(d != 0 && d < t) {
			t = d;
			id = i;
		}
		SYCL_END
	}
	SYCL_END

	return t < inf;
}

// http://stackoverflow.com/a/16077942
float1 getRandom(uint2& seed) {
	// Note: Should not be declared static
	const float1 invMaxInt = 1.0f / 4294967296.0f;
	uint1 x = seed.x * 17 + seed.y * 13123;
	seed.x = (x << 13) ^ x;
	seed.y ^= x << 7;
	return (float1)(x * (x * x * 15731 + 74323) + 871483) * invMaxInt;
}

void radiance(
	Vector& return_,
	spheres_t spheres,
	RaySycl r,
	uint2& randomSeed,
	Vector cl = { 0, 0, 0 },
	Vector cf = { 1, 1, 1 }
) {
	using namespace cl::sycl;

	float1 t;	// distance to intersection
	int1 id = 0;	// id of intersected object
	int1 depth = 0;

	// cl is accumulated color
	// cf is accumulated reflectance

	RaySycl reflRay(0, 0);
	Vector x, tdir;

	SYCL_WHILE(true) {
		SYCL_IF(!intersect(spheres, r, t, id)) {
			// if miss, don't add anything
			return_ = cl;
			SYCL_BREAK
		}
		SYCL_END

		SphereSycl obj(spheres[id]); // the hit object
		x = r.o + r.d*t;

		Vector n = Vector(x - obj.p()).norm();
		Vector nl = n;
		SYCL_IF(n.dot(r.d) > 0)
			nl *= -1;
		SYCL_END

		Vector f = obj.c();

		float1 p;	// max refl
		SYCL_IF(f.x > f.y && f.x > f.z)
			p = f.x;
		SYCL_ELSE_IF(f.y > f.z)
			p = f.y;
		SYCL_ELSE
			p = f.z;
		SYCL_END

		cl += cf.mult(obj.e());

		depth += 1; 
		SYCL_IF(depth > 5) {
			SYCL_IF(getRandom(randomSeed) < p) {
				f *= 1 / p;
			}
			SYCL_ELSE {
				return_ = cl;
				SYCL_BREAK
			}
			SYCL_END
		}
		SYCL_END

		cf = cf.mult(f);

		SYCL_IF(obj.refl() == (float)DIFF) { // Ideal DIFFUSE reflection 
			float1 r1 = 2 * M_PI * getRandom(randomSeed);
			float1 r2 = getRandom(randomSeed);
			float1 r2s = sqrt(r2);
			Vector w = nl;
			
			Vector u(0, 0, 0);
			SYCL_IF(fabs(w.x) > .1f)
				u.y = 1;
			SYCL_ELSE
				u.x = 1;
			SYCL_END
			u = (u % w).norm();

			Vector v = w % u;
			Vector d = Vector(u*cos(r1)*r2s + v*sin(r1)*r2s + w*sqrt(1 - r2)).norm();

			// Recursion
			r = RaySycl(x, d);
			SYCL_CONTINUE
		}
		SYCL_ELSE_IF(obj.refl() == (float)SPEC) { // Ideal SPECULAR reflection 
			// Recursion
			r = RaySycl(x, r.d - n * 2 * n.dot(r.d));
			SYCL_CONTINUE
		}
		SYCL_END

		reflRay = RaySycl(x, r.d - n * 2 * n.dot(r.d));	// Ideal dielectric REFRACTION
		bool1 into = n.dot(nl) > 0;	// Ray from outside going in?
		float1 nc = 1;
		float1 nt = 1.5;

		float1 nnt;
		SYCL_IF(into)
			nnt = nc / nt;
		SYCL_ELSE
			nnt = nt / nc;
		SYCL_END

		float1 ddn = r.d.dot(nl);
		float1 cos2t = 1 - nnt*nnt*(1 - ddn*ddn);
		SYCL_IF(cos2t < 0) { // Total internal reflection 	
			// Recursion
			r = reflRay;
			SYCL_CONTINUE
		}
		SYCL_END

		float1 tmp = 1;
		SYCL_IF(!into)
			tmp = -1;
		SYCL_END

		tdir = Vector(r.d*nnt - n*(tmp*(ddn*nnt + sqrt(cos2t)))).norm();
		float1 a = nt - nc;
		float1 b = nt + nc;
		float1 R0 = a*a / (b*b);

		float1 c = 1;
		SYCL_IF(into)
			c += ddn;
		SYCL_ELSE
			c -= tdir.dot(n);
		SYCL_END

		float1 Re = R0 + (1 - R0)*c*c*c*c*c;
		float1 Tr = 1 - Re;
		float1 P = .25 + .5*Re;
		float1 RP = Re / P;
		float1 TP = Tr / (1 - P);

		// Tail recursion
		SYCL_IF(getRandom(randomSeed) < P) {
			cf *= RP;
			r = reflRay;
		}
		SYCL_ELSE {
			cf *= TP;
			r = RaySycl(x, tdir);
		}
		SYCL_END
	}
	SYCL_END
}

template <class T, class D>
void assign(T& target, D& data) {
	target.x = data.x;
	target.y = data.y;
	target.z = data.z;
}

} // ns_sycl_gtx

void compute_sycl_gtx(void* dev, int w, int h, int samps, Ray& cam_, Vec& cx_, Vec& cy_, Vec r_, Vec* c_) {
	using namespace std;
	using namespace cl::sycl;
	using namespace ns_sycl_gtx;

	queue q(*(device*)dev);

	unsigned short Xi[3] = { 0, 0, 0 };

	buffer<float3> colorsBuffer(range<1>(w*h));

	vector<cl_uint2> seedArray;
	seedArray.reserve(w*h);
	for(int y = 0; y < h; ++y) {
		Xi[2] = y*y*y;
		for(int x = 0; x < w; ++x) {
			seedArray.push_back({ erand48(Xi), erand48(Xi) });
		}
	}
	buffer<cl_uint2> seedsBuffer(seedArray);

	buffer<float16> spheres_(ns_sycl_gtx::numSpheres);
	{
		auto s = spheres_.get_access<access::discard_write, access::host_buffer>();
		// See SphereSycl
		for(int i = 0; i < ns_sycl_gtx::numSpheres; ++i) {
			auto& si = s[i];
			auto& sj = ns_sycl_gtx::spheres[i];

			assign(si.lo.lo, sj.p);
			assign(si.lo.hi, sj.e);
			assign(si.hi.lo, sj.c);

			si.lo.lo.w = sj.rad;
			si.hi.lo.w = sj.refl;
		}
	}

	// Divide GPU calculation into parts to prevent monitor freezing
	const int numParts =
		(q.get_device().get_info<info::device::device_type>() == info::device_type::gpu)
			? std::min(samps / 5 + 1, h / 10)
			: 1;
	vector<decltype(colorsBuffer)> colors;
	vector<decltype(seedsBuffer)> seeds_;
	vector<pair<int, int>> lineOffset;
	for(int k = 0; k < numParts; ++k) {
		int start = h*k / numParts;
		int end = h*(k + 1) / numParts;
		int height = end - start;
		range<1> length(height*w);

		lineOffset.emplace_back(start, height);
		colors.emplace_back(colorsBuffer, id<1>(start*w), length);
		seeds_.emplace_back(seedsBuffer, id<1>(start*w), length);
	}

	for(auto k = 0; k < numParts; ++k) {
		q.submit([&](handler& cgh) {
			auto c = colors[k].get_access<access::discard_read_write, access::global_buffer>(cgh);

			// TODO: constant_buffer
			auto spheres = spheres_.get_access<access::read, access::global_buffer>(cgh);
			auto seeds = seeds_[k].get_access<access::read, access::global_buffer>(cgh);

			cgh.parallel_for<class smallpt>(range<2>(w, lineOffset[k].second), [=](id<2> i) {
				Vector cx(cx_);
				Vector cy(cy_);
				Vector r(r_);
				RaySycl cam(cam_);
				uint2 randomSeed;
				randomSeed = seeds[i] * (i + 1) + i + 1;

				c[i] = 0; // Important to start at zero

				// 2x2 subpixel rows
				SYCL_FOR(int1 sy = 0, sy < 2, sy++) {
					// 2x2 subpixel cols
					SYCL_FOR(int1 sx = 0, sx < 2, sx++) {
						SYCL_FOR(int1 s = 0, s < samps, s++) {
							float2 rnew;
							rnew.x = 2 * getRandom(randomSeed);
							rnew.y = 2 * getRandom(randomSeed);

							float2 dd;

							SYCL_IF(rnew.x < 1)
								dd.x = sqrt(rnew.x) - 1;
							SYCL_ELSE
								dd.x = 1 - sqrt(2 - rnew.x);
							SYCL_END

							SYCL_IF(rnew.y < 1)
								dd.y = sqrt(rnew.y) - 1;
							SYCL_ELSE
								dd.y = 1 - sqrt(2 - rnew.y);
							SYCL_END

							Vector d =
								cx * (((sx + .5f + dd.x) / 2 + i[0]) / w - .5f) +
								cy * (((sy + .5f + dd.y) / 2 + i[1] + lineOffset[k].first) / h - .5f) +
								cam.d;

							// TODO:
							Vector rad;
							radiance(rad, spheres, RaySycl(cam.o + d * 140, d.norm()), randomSeed);
							r += rad*(1.f / samps);
						} // Camera rays are pushed ^^^^^ forward to start in interior
						SYCL_END

						clamp(r.x);
						clamp(r.y);
						clamp(r.z);

						c[i] += r * .25f;

						r = Vector();
					}
					SYCL_END
				}
				SYCL_END
			});
		});
	}

	for(auto k = 0; k < numParts; ++k) {
#if _DEBUG
		cout << "Waiting for kernel to finish ..." << endl;
#endif
		auto c = colors[k].get_access<access::read, access::host_buffer>();

#if _DEBUG
		cout << "Copying results ..." << endl;
#endif
		auto start = lineOffset[k].first;
		auto end = start + lineOffset[k].second;
		for(int y = start; y < end; ++y) {
			int i = (y - start)*w;
			int ri = (h - y - 1) * w;	// Picture is upside down
			for(int x = 0; x < w; ++x) {
				auto& ci = c[i];
				assign(c_[ri], ci);

				++i;
				++ri;
			}
		}
	}
}

void compute_sycl_gtx_cpu(void*, int w, int h, int samps, Ray& cam, Vec& cx, Vec& cy, Vec r, Vec* c) {
	cl::sycl::cpu_selector cpu;
	cl::sycl::device dev(cpu);
	compute_sycl_gtx(&dev, w, h, samps, cam, cx, cy, r, c);
}

void compute_sycl_gtx_gpu(void*, int w, int h, int samps, Ray& cam, Vec& cx, Vec& cy, Vec r, Vec* c) {
	cl::sycl::gpu_selector gpu;
	cl::sycl::device dev(gpu);
	compute_sycl_gtx(&dev, w, h, samps, cam, cx, cy, r, c);
}
