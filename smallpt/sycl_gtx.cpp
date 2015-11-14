// smallpt, a Path Tracer by Kevin Beason, 2008
//
// Modified by Peter Žužek
// For the original code, see github.com/munificient/smallpt
// For the original license, see smallpt.LICENSE.txt

#include <cmath>
#include <cstdlib>
#include <cstdio>

#include "classes.h"
#include "msvc.h"

namespace ns_sycl_gtx {

Sphere spheres[] = {//Scene: radius, position, emission, color, material
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

inline bool intersect(const Ray &r, double &t, int &id) {
	double n = sizeof(spheres) / sizeof(Sphere);
	double d;
	double inf = t = 1e20;
	for(int i = int(n); i--;) {
		if((d = spheres[i].intersect(r)) && d < t) {
			t = d;
			id = i;
		}
	}
	return t < inf;
}

Vec radiance(const Ray &r, int depth, unsigned short *Xi) {
	double t;								// distance to intersection
	int id = 0;								// id of intersected object
	if(!intersect(r, t, id)) {
		return Vec(); // if miss, return black
	}
	const Sphere &obj = spheres[id];		// the hit object
	Vec x = r.o + r.d*t;
	Vec n = (x - obj.p).norm();
	Vec nl = n.dot(r.d) < 0 ? n : n*-1;
	Vec f = obj.c;
	double p = f.x > f.y && f.x>f.z ? f.x : f.y > f.z ? f.y : f.z; // max refl

	if(++depth > 5) {
		if(erand48(Xi) < p) {
			f = f*(1 / p);
		}
		else {
			return obj.e;
		}
	}

	if(obj.refl == DIFF) {					// Ideal DIFFUSE reflection
		double r1 = 2 * M_PI*erand48(Xi), r2 = erand48(Xi), r2s = sqrt(r2);
		Vec w = nl;
		Vec u = ((fabs(w.x) > .1 ? Vec(0, 1) : Vec(1)) % w).norm();
		Vec v = w%u;
		Vec d = (u*cos(r1)*r2s + v*sin(r1)*r2s + w*sqrt(1 - r2)).norm();
		return obj.e + f.mult(radiance(Ray(x, d), depth, Xi));
	}
	else if(obj.refl == SPEC) {			// Ideal SPECULAR reflection
		return obj.e + f.mult(radiance(Ray(x, r.d - n * 2 * n.dot(r.d)), depth, Xi));
	}
	Ray reflRay(x, r.d - n * 2 * n.dot(r.d));	 // Ideal dielectric REFRACTION
	bool into = n.dot(nl) > 0;				// Ray from outside going in?
	double nc = 1;
	double nt = 1.5;
	double nnt = into ? nc / nt : nt / nc;
	double ddn = r.d.dot(nl);
	double cos2t;
	if((cos2t = 1 - nnt*nnt*(1 - ddn*ddn)) < 0) {	// Total internal reflection
		return obj.e + f.mult(radiance(reflRay, depth, Xi));
	}
	Vec tdir = (r.d*nnt - n*((into ? 1 : -1)*(ddn*nnt + sqrt(cos2t)))).norm();
	double a = nt - nc;
	double b = nt + nc;
	double R0 = a*a / (b*b);
	double c = 1 - (into ? -ddn : tdir.dot(n));
	double Re = R0 + (1 - R0)*c*c*c*c*c;
	double Tr = 1 - Re, P = .25 + .5*Re;
	double RP = Re / P, TP = Tr / (1 - P);
	return obj.e + f.mult(depth > 2 ? (erand48(Xi) < P ?	// Russian roulette
		radiance(reflRay, depth, Xi)*RP : radiance(Ray(x, tdir), depth, Xi)*TP) :
		radiance(reflRay, depth, Xi)*Re + radiance(Ray(x, tdir), depth, Xi)*Tr);
}
} // namespace ns_sycl_gtx

inline double clamp(double x) {
	return x < 0 ? 0 : x>1 ? 1 : x;
}

inline int toInt(double x) {
	return int(pow(clamp(x), 1 / 2.2) * 255 + .5);
}

int sycl_gtx(int argc, char *argv[]) {
	int w = 1024, h = 768, samps = argc == 2 ? atoi(argv[1]) / 4 : 1; // # samples
	Ray cam(Vec(50, 52, 295.6), Vec(0, -0.042612, -1).norm()); // cam pos, dir
	Vec cx = Vec(w*.5135 / h), cy = (cx%cam.d).norm()*.5135, r, *c = new Vec[w*h];
	for(int y = 0; y < h; y++) {						// Loop over image rows
		fprintf(stderr, "\rRendering (%d spp) %5.2f%%", samps * 4, 100.*y / (h - 1));
		for(unsigned short x = 0, Xi[3] = { 0, 0, y*y*y }; x < w; x++) {	// Loop cols
			for(int sy = 0, i = (h - y - 1)*w + x; sy < 2; sy++) {	 // 2x2 subpixel rows
				for(int sx = 0; sx < 2; sx++, r = Vec()) {		// 2x2 subpixel cols
					for(int s = 0; s < samps; s++) {
						double r1 = 2 * erand48(Xi);
						double dx = r1 < 1 ? sqrt(r1) - 1 : 1 - sqrt(2 - r1);

						double r2 = 2 * erand48(Xi);
						double dy = r2 < 1 ? sqrt(r2) - 1 : 1 - sqrt(2 - r2);

						Vec d = cx*(((sx + .5 + dx) / 2 + x) / w - .5) +
							cy*(((sy + .5 + dy) / 2 + y) / h - .5) + cam.d;
						r = r + ns_sycl_gtx::radiance(Ray(cam.o + d * 140, d.norm()), 0, Xi)*(1. / samps);
					} // Camera rays are pushed ^^^^^ forward to start in interior

					c[i] = c[i] + Vec(clamp(r.x), clamp(r.y), clamp(r.z))*.25;
				}
			}
		}
	}
	FILE *f = fopen("image.ppm", "w");		 // Write image to PPM file.
	fprintf(f, "P3\n%d %d\n%d\n", w, h, 255);
	for(int i = 0; i < w*h; i++) {
		fprintf(f, "%d %d %d ", toInt(c[i].x), toInt(c[i].y), toInt(c[i].z));
	}
	return 0;
}
