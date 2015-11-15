// smallpt, a Path Tracer by Kevin Beason, 2008
//
// Modified by Peter Žužek
// For the original code, see github.com/munificient/smallpt
// For the original license, see smallpt.LICENSE.txt

#include <cmath>
#include <cstdlib>
#include <cstdio>

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
void compute_sycl_gtx(int w, int h, int samps, Ray& cam, Vec& cx, Vec& cy, Vec r, Vec* c, cl::sycl::device_selector& selector) {
	using namespace cl::sycl;

	queue q(selector);

	buffer<Vec> colors(c, range<1>(w*h));

	q.submit([&](handler& cgh) {
		// TODO
	});
}

void compute_sycl_gtx_cpu(int w, int h, int samps, Ray& cam, Vec& cx, Vec& cy, Vec r, Vec* c) {
	cl::sycl::cpu_selector cpu;
	compute_sycl_gtx(w, h, samps, cam, cx, cy, r, c, cpu);
}

void compute_sycl_gtx_gpu(int w, int h, int samps, Ray& cam, Vec& cx, Vec& cy, Vec r, Vec* c) {
	cl::sycl::gpu_selector gpu;
	compute_sycl_gtx(w, h, samps, cam, cx, cy, r, c, gpu);
}
