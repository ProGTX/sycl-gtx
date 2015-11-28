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

static const int numSpheres = 9;
Sphere spheres[numSpheres] = {//Scene: radius, position, emission, color, material
	Sphere(1e5, Vec(1e5 + 1, 40.8, 81.6), Vec(), Vec(.75, .25, .25), DIFF),//Left
	Sphere(1e5, Vec(-1e5 + 99, 40.8, 81.6), Vec(), Vec(.25, .25, .75), DIFF),//Rght
	Sphere(1e5, Vec(50, 40.8, 1e5), Vec(), Vec(.75, .75, .75), DIFF),//Back
	Sphere(1e5, Vec(50, 40.8, -1e5 + 170), Vec(), Vec(), DIFF),//Frnt
	Sphere(1e5, Vec(50, 1e5, 81.6), Vec(), Vec(.75, .75, .75), DIFF),//Botm
	Sphere(1e5, Vec(50, -1e5 + 81.6, 81.6), Vec(), Vec(.75, .75, .75), DIFF),//Top
	Sphere(16.5, Vec(27, 16.5, 47), Vec(), Vec(1, 1, 1)*.999, SPEC),//Mirr
	Sphere(16.5, Vec(73, 16.5, 78), Vec(), Vec(1, 1, 1)*.999, REFR),//Glas
	Sphere(600, Vec(50, 681.6 - .27, 81.6), Vec(12, 12, 12), Vec(), DIFF) //Lite
};

inline bool intersect(const Ray& r, double& t, int& id) {
	double d;
	double inf = t = 1e20;
	for(int i = numSpheres; i > 0;) {
		--i;
		if((d = spheres[i].intersect(r)) && d < t) {
			t = d;
			id = i;
		}
	}
	return t < inf;
}

enum class DoNext {
	Return, ContinueLoop, Proceed
};

DoNext radianceInner(
	Ray& r, int& depth, unsigned short* Xi,	// Original parameters
	double& t, int& id, Vec& cl, Vec& cf,	// Passed references
	// Output references
	double& Re, double& Tr, double& P, double& RP, double& TP, Ray& reflRay, Vec& x, Vec& tdir
) {
	if(!intersect(r, t, id)) {
		// if miss, don't add anything
		return DoNext::Return;
	}
	const Sphere& obj = spheres[id]; // the hit object
	x = r.o + r.d*t;
	Vec n = (x - obj.p).norm();
	Vec nl = n;
	if(n.dot(r.d) > 0) {
		nl = nl * -1;
	}
	Vec f = obj.c;
	double p;	// max refl
	if(f.x > f.y && f.x > f.z) {
		p = f.x;
	}
	else if(f.y > f.z) {
		p = f.y;
	}
	else {
		p = f.z;
	}

	cl = cl + cf.mult(obj.e);

	depth += 1;
	if(depth > 5) {
		if(erand48(Xi) < p) {
			f = f*(1 / p);
		}
		else {
			return DoNext::Return;
		}
	}

	cf = cf.mult(f);

	if(obj.refl == DIFF) {	// Ideal DIFFUSE reflection
		double r1 = 2 * M_PI*erand48(Xi), r2 = erand48(Xi), r2s = sqrt(r2);
		Vec w = nl;
		Vec u;
		if(fabs(w.x) > .1) {
			u.y = 1;
		}
		else {
			u.x = 1;
		}
		u = (u % w).norm();
		Vec v = w%u;
		Vec d = (u*cos(r1)*r2s + v*sin(r1)*r2s + w*sqrt(1 - r2)).norm();

		// Recursion
		r = Ray(x, d);
		return DoNext::ContinueLoop;
	}
	else if(obj.refl == SPEC) {	// Ideal SPECULAR reflection
		// Recursion
		r = Ray(x, r.d - n * 2 * n.dot(r.d));
		return DoNext::ContinueLoop;
	}
	reflRay = Ray(x, r.d - n * 2 * n.dot(r.d));	// Ideal dielectric REFRACTION
	bool into = n.dot(nl) > 0;	// Ray from outside going in?
	double nc = 1;
	double nt = 1.5;
	double nnt;
	if(into) {
		nnt = nc / nt;
	}
	else {
		nnt = nt / nc;
	}
	double ddn = r.d.dot(nl);
	double cos2t;
	if((cos2t = 1 - nnt*nnt*(1 - ddn*ddn)) < 0) {	// Total internal reflection
		// Recursion
		r = reflRay;
		return DoNext::ContinueLoop;
	}
	double tmp = 1; 
	if(!into) {
		tmp = -1;
	}
	tdir = (r.d*nnt - n*(tmp*(ddn*nnt + sqrt(cos2t)))).norm();
	double a = nt - nc;
	double b = nt + nc;
	double R0 = a*a / (b*b);
	double c = 1;
	if(into) {
		c += ddn;
	}
	else {
		c -= tdir.dot(n);
	}
	Re = R0 + (1 - R0)*c*c*c*c*c;
	Tr = 1 - Re;
	P = .25 + .5*Re;
	RP = Re / P;
	TP = Tr / (1 - P);

	return DoNext::Proceed;
}

void radiance(Ray r, int depth, unsigned short* Xi, Vec& cl, Vec& cf) {
	double t;	// distance to intersection
	int id = 0;	// id of intersected object

	double Re, Tr, P, RP, TP;
	Ray reflRay(0, 0);
	Vec x, tdir;

	while(true) {
		auto doNext = radianceInner(
			r, depth, Xi,
			t, id, cl, cf,
			Re, Tr, P, RP, TP, reflRay, x, tdir
		);

		if(doNext == DoNext::ContinueLoop) {
			continue;
		}
		if(doNext == DoNext::Return) {
			return;
		}

		if(erand48(Xi) < P) {
			cf = cf * RP;
			r = reflRay;
		}
		else {
			cf = cf * TP;
			r = Ray(x, tdir);
		}
	}
}

template <int depth_ = 0>
Vec radiance(Ray r, unsigned short* Xi, Vec cl = { 0, 0, 0 }, Vec cf = {1, 1, 1}) {
	double t;	// distance to intersection
	int id = 0;	// id of intersected object
	int depth = depth_;

	// cl is accumulated color
	// cf is accumulated reflectance

	double Re, Tr, P, RP, TP;
	Ray reflRay(0, 0);
	Vec x, tdir;

	while(true) {
		auto doNext = radianceInner(
			r, depth, Xi,
			t, id, cl, cf,
			Re, Tr, P, RP, TP, reflRay, x, tdir
		);

		if(doNext == DoNext::ContinueLoop) {
			continue;
		}
		if(doNext == DoNext::Return) {
			return cl;
		}

		if(depth == 1) {
			return radiance<1>(reflRay, Xi, cl, cf * Re) + radiance<1>(Ray(x, tdir), Xi, cl, cf * Tr);
		}
		else if(depth == 2) {
			return radiance<2>(reflRay, Xi, cl, cf * Re) + radiance<2>(Ray(x, tdir), Xi, cl, cf * Tr);
		}
		else {
			radiance(r, depth, Xi, cl, cf);
			return cl;
		}
	}
}

} // namespace ns_sycl_gtx

inline double clamp(double x) {
	if(x < 0) {
		return 0;
	}
	if(x > 1) {
		return 1;
	}
	return x;
}

void compute_sycl_gtx_openmp(int w, int h, int samps, Ray& cam, Vec& cx, Vec& cy, Vec r, Vec* c) {
	#pragma omp parallel for schedule(dynamic, 1) private(r)
	for(int y = 0; y < h; y++) {						// Loop over image rows
		fprintf(stderr, "\rRendering (%d spp) %5.2f%%", samps * 4, 100.*y / (h - 1));
		for(unsigned short x = 0, Xi[3] = { 0, 0, y*y*y }; x < w; x++) {	// Loop cols
			for(int sy = 0, i = (h - y - 1)*w + x; sy < 2; sy++) {	 // 2x2 subpixel rows
				for(int sx = 0; sx < 2; sx++, r = Vec()) {		// 2x2 subpixel cols
					for(int s = 0; s < samps; s++) {
						double r1 = 2 * erand48(Xi);
						double r2 = 2 * erand48(Xi);

						double dx, dy;
						if(r1 < 1) {
							dx = sqrt(r1) - 1;
						}
						else {
							dx = 1 - sqrt(2 - r1);
						}
						if(r2 < 1) {
							dy = sqrt(r2) - 1;
						}
						else {
							dy = 1 - sqrt(2 - r2);
						}

						Vec d = cx*(((sx + .5 + dx) / 2 + x) / w - .5) + cy*(((sy + .5 + dy) / 2 + y) / h - .5) + cam.d;
						r = r + ns_sycl_gtx::radiance(Ray(cam.o + d * 140, d.norm()), Xi)*(1. / samps);
					} // Camera rays are pushed ^^^^^ forward to start in interior

					c[i] = c[i] + Vec(clamp(r.x), clamp(r.y), clamp(r.z))*.25;
				}
			}
		}
	}
}

namespace sycl_class {

using cl::sycl::bool1;
using cl::sycl::int1;
using cl::sycl::uint1;
using cl::sycl::uint2;
using cl::sycl::float1;
using cl::sycl::double1;
using cl::sycl::double3;
using cl::sycl::double16;
using spheres_t = cl::sycl::accessor<double16, 1, cl::sycl::access::read, cl::sycl::access::global_buffer>;

#ifdef SYCL_GTX
struct Vector : public double3 {
	Vector()
		: double3(0, 0, 0) {}
	Vector(const Vec& v)
		: double3(v.x, v.y, v.z) {}
	template <class T>
	Vector(T n)
		: double3(n) {}
	template <class X, class Y, class Z>
	Vector(X&& x, Y&& y, Z&& z)
		: double3(x, y, z) {}

	template <class T>
	Vector& operator=(T&& t) {
		double3::operator=(t);
		return *this;
	}

	Vector mult(const Vector &b) const {
		return Vector(x*b.x, y*b.y, z*b.z);
	}
	Vector& norm() {
		operator*=(1 / cl::sycl::sqrt(x*x + y*y + z*z));
		return *this;
	}
	double1 dot(const Vector &b) const {
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

inline void clamp(double1& x) {
	SYCL_IF(x < 0)
	SYCL_BEGIN {
		x = 0;
	}
	SYCL_END
	SYCL_ELSE_IF(x > 1)
	SYCL_BEGIN {
		x = 1;
	}
	SYCL_END
}


struct SphereSycl {
private:
	double16 data;

public:
	template <class T>
	SphereSycl(T&& data)
		: data(data) {}

	double1& rad() const {
		return data.w;
	}
	double3& p() const { // position
		return data.xyz;
	}
	double3& e() { // emission
		return data.lo.hi.xyz;
	}
	double3& c() const { // color
		return data.hi.xyz;
	}
	double1 refl() const { // reflection type (Refl_t)
		return data.hi.w;
	}

	void intersect(double1& return_, const RaySycl& r) const { // returns distance, 0 if no hit
		Vector op = p() - r.o; // Solve t^2*d.d + 2*t*(o-p).d + (o-p).(o-p)-R^2 = 0
		double1 t;
		double1 eps = 1e-4;
		double1 b = op.dot(r.d);
		double1 det = b*b - op.dot(op) + rad()*rad();

		SYCL_IF(det < 0)
		SYCL_BEGIN {
			return_ = 0;
		}
		SYCL_END
		SYCL_ELSE
		SYCL_BEGIN {
			det = sqrt(det);
			t = b - det;
			SYCL_IF(t > eps)
			SYCL_BEGIN {
				return_ = t;
			}
			SYCL_END
			SYCL_ELSE
			SYCL_BEGIN{
				t = b + det;
				SYCL_IF(t > eps)
				SYCL_BEGIN {
					return_ = t;
				}
				SYCL_END
				SYCL_ELSE
				SYCL_BEGIN {
					return_ = 0;
				}
				SYCL_END
			}
			SYCL_END
		}
		SYCL_END
	}
};


inline bool1 intersect(spheres_t spheres, const RaySycl& r, double1& t, int1& id) {
	using namespace cl::sycl;
	double1 d;
	double1 inf = t = 1e20;

	int1 i = ns_sycl_gtx::numSpheres;
	SYCL_WHILE(i > 0)
	SYCL_BEGIN {
		i -= 1;
		SphereSycl(spheres[i]).intersect(d, r);
		SYCL_IF(d != 0 && d < t)
		SYCL_BEGIN {
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
	static const float1 invMaxInt = 1.0f / 4294967296.0f;
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

	double1 t;	// distance to intersection
	int1 id = 0;	// id of intersected object
	int1 depth = 0;

	// cl is accumulated color
	// cf is accumulated reflectance

	RaySycl reflRay(0, 0);
	Vector x, tdir;

	SYCL_WHILE(true)
	SYCL_BEGIN {
		SYCL_IF(!intersect(spheres, r, t, id))
		SYCL_BEGIN {
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
		SYCL_BEGIN {
			nl *= -1;
		}
		SYCL_END

		Vector f = obj.c();

		double1 p;	// max refl
		SYCL_IF(f.x > f.y && f.x > f.z)
		SYCL_BEGIN {
			p = f.x;
		}
		SYCL_END
		SYCL_ELSE_IF(f.y > f.z)
		SYCL_BEGIN {
			p = f.y;
		}
		SYCL_END
		SYCL_ELSE
		SYCL_BEGIN {
			p = f.z;
		}
		SYCL_END

		cl += cf.mult(obj.e());

		depth += 1; 
		SYCL_IF(depth > 5)
		SYCL_BEGIN {
			SYCL_IF(getRandom(randomSeed) < p)
			SYCL_BEGIN {
				f *= 1 / p;
			}
			SYCL_END
			SYCL_ELSE
			SYCL_BEGIN {
				return_ = cl;
				SYCL_BREAK
			}
			SYCL_END
		}
		SYCL_END

		cf = cf.mult(f);

		SYCL_IF(obj.refl() == (double)DIFF) // Ideal DIFFUSE reflection
		SYCL_BEGIN {
			double1 r1 = 2 * M_PI * getRandom(randomSeed);
			double1 r2 = getRandom(randomSeed);
			double1 r2s = sqrt(r2);
			Vector w = nl;
			
			Vector u(0, 0, 0);
			SYCL_IF(fabs(w.x) > .1)
			SYCL_BEGIN {
				u.y = 1;
			}
			SYCL_END
			SYCL_ELSE
			SYCL_BEGIN {
				u.x = 1;
			}
			SYCL_END
			u = (u % w).norm();

			Vector v = w % u;
			Vector d = Vector(u*cos(r1)*r2s + v*sin(r1)*r2s + w*sqrt(1 - r2)).norm();

			// Recursion
			r = RaySycl(x, d);
			SYCL_CONTINUE
		}
		SYCL_END
		SYCL_ELSE_IF(obj.refl() == (double)SPEC)// Ideal SPECULAR reflection
		SYCL_BEGIN {
			// Recursion
			r = RaySycl(x, r.d - n * 2 * n.dot(r.d));
			SYCL_CONTINUE
		}
		SYCL_END

		reflRay = RaySycl(x, r.d - n * 2 * n.dot(r.d));	// Ideal dielectric REFRACTION
		bool1 into = n.dot(nl) > 0;	// Ray from outside going in?
		double1 nc = 1;
		double1 nt = 1.5;

		double1 nnt;
		SYCL_IF(into)
		SYCL_BEGIN {
			nnt = nc / nt;
		}
		SYCL_END
		SYCL_ELSE
		SYCL_BEGIN {
			nnt = nt / nc;
		}
		SYCL_END

		double1 ddn = r.d.dot(nl);
		double1 cos2t = 1 - nnt*nnt*(1 - ddn*ddn);
		SYCL_IF(cos2t < 0) // // Total internal reflection
		SYCL_BEGIN {	
			// Recursion
			r = reflRay;
			SYCL_CONTINUE
		}
		SYCL_END

		double1 tmp = 1;
		SYCL_IF(!into)
		SYCL_BEGIN {
			tmp = -1;
		}
		SYCL_END

		tdir = Vector(r.d*nnt - n*(tmp*(ddn*nnt + sqrt(cos2t)))).norm();
		double1 a = nt - nc;
		double1 b = nt + nc;
		double1 R0 = a*a / (b*b);

		double1 c = 1;
		SYCL_IF(into)
		SYCL_BEGIN {
			c += ddn;
		}
		SYCL_END
		SYCL_ELSE
		SYCL_BEGIN {
			c -= tdir.dot(n);
		}
		SYCL_END

		double1 Re = R0 + (1 - R0)*c*c*c*c*c;
		double1 Tr = 1 - Re;
		double1 P = .25 + .5*Re;
		double1 RP = Re / P;
		double1 TP = Tr / (1 - P);

		// Tail recursion
		SYCL_IF(getRandom(randomSeed) < P)
		SYCL_BEGIN {
			cf *= RP;
			r = reflRay;
		}
		SYCL_END
		SYCL_ELSE
		SYCL_BEGIN {
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

} // sycl_class

void compute_sycl_gtx(int w, int h, int samps, Ray& cam_, Vec& cx_, Vec& cy_, Vec r_, Vec* c_, cl::sycl::device_selector& selector) {
	using namespace std;
	using namespace cl::sycl;
	using sycl_class::Vector;
	using sycl_class::RaySycl;
	using sycl_class::assign;
	using sycl_class::getRandom;

	queue q(selector);

	unsigned short Xi[3] = { 0, 0, 0 };

	buffer<double3> colors(range<1>(w*h));
	buffer<double16> spheres_(ns_sycl_gtx::numSpheres);
	buffer<uint2> seeds_(w*h);
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

		auto c = colors.get_access<access::discard_write, access::host_buffer>();
		auto rs = seeds_.get_access<access::discard_write, access::host_buffer>();
		for(int y = 0; y < h; ++y) {
			for(int x = 0; x < w; ++x) {
				int i = y*w + x;
				
				auto& ci = c[i];
				assign(ci, c_[i]);
				Xi[2] = y*y*y;

				auto& rsi = rs[i];
				rsi.x = erand48(Xi);
				rsi.y = erand48(Xi);
			}
		}
	}

	q.submit([&](handler& cgh) {
		auto c = colors.get_access<access::read_write, access::global_buffer>(cgh);

		// TODO: constant_buffer
		auto spheres = spheres_.get_access<access::read, access::global_buffer>(cgh);
		auto seeds = seeds_.get_access<access::read, access::global_buffer>(cgh);

		cgh.parallel_for<class smallpt>(range<2>(w, h), [=](id<2> i) {
			Vector cx(cx_);
			Vector cy(cy_);
			Vector r(r_);
			RaySycl cam(cam_);
			uint2 randomSeed;
			randomSeed = seeds[i] * (i + 1) + i;

			// 2x2 subpixel rows
			SYCL_FOR(int1 sy = 0, sy < 2, sy++)
			SYCL_BEGIN {
				// 2x2 subpixel cols
				SYCL_FOR(int1 sx = 0, sx < 2, sx++)
				SYCL_BEGIN {
					SYCL_FOR(int1 s = 0, s < samps, s++)
					SYCL_BEGIN {
						double2 rnew;
						rnew.x = 2 * getRandom(randomSeed);
						rnew.y = 2 * getRandom(randomSeed);

						double2 dd;

						SYCL_IF(rnew.x < 1)
						SYCL_BEGIN {
							dd.x = sqrt(rnew.x) - 1;
						}
						SYCL_END
						SYCL_ELSE
						SYCL_BEGIN {
							dd.x = 1 - sqrt(2 - rnew.x);
						}
						SYCL_END

						SYCL_IF(rnew.y < 1)
						SYCL_BEGIN {
							dd.y = sqrt(rnew.y) - 1;
						}
						SYCL_END
						SYCL_ELSE
						SYCL_BEGIN {
							dd.y = 1 - sqrt(2 - rnew.y);
						}
						SYCL_END

						Vector d =
							cx * (((sx + .5 + dd.x) / 2 + i[0]) / w - .5) +
							cy * (((sy + .5 + dd.y) / 2 + i[1]) / h - .5) +
							cam.d;

						// TODO:
						Vector rad;
						sycl_class::radiance(rad, spheres, RaySycl(cam.o + d * 140, d.norm()), randomSeed);
						r += rad*(1. / samps);
					} // Camera rays are pushed ^^^^^ forward to start in interior
					SYCL_END

					sycl_class::clamp(r.x);
					sycl_class::clamp(r.y);
					sycl_class::clamp(r.z);

					c[i] += r * .25;

					r = Vector();
				}
				SYCL_END
			}
			SYCL_END
		});
	});

	auto c = colors.get_access<access::read, access::host_buffer>();
	for(int y = 0; y < h; ++y) {
		for(int x = 0; x < w; ++x) {
			int i = y*w + x;
			auto& ci = c[i];
			assign(c_[i], ci);
		}
	}
}

void compute_sycl_gtx_cpu(int w, int h, int samps, Ray& cam, Vec& cx, Vec& cy, Vec r, Vec* c) {
	cl::sycl::cpu_selector cpu;
	compute_sycl_gtx(w, h, samps, cam, cx, cy, r, c, cpu);
}

void compute_sycl_gtx_gpu(int w, int h, int samps, Ray& cam, Vec& cx, Vec& cy, Vec r, Vec* c) {
	cl::sycl::gpu_selector gpu;
	compute_sycl_gtx(w, h, samps, cam, cx, cy, r, c, gpu);
}
