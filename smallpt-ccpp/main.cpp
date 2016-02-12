/******************************************************************************
*
* Modified version for SYCL of Kevin Beason smallpt
* http://www.kevinbeason.com/smallpt/
*
******************************************************************************/

#define float_type float
#define sqrt_f cl::sycl::sqrt
#include <smallpt.h>
#include <sycl_gtx.h>

using namespace cl;
using namespace sycl;

class RNG {
public:
	unsigned int x;
	const uint32_t fmask = (1 << 23) - 1;
	RNG(const unsigned int seed) { x = seed; }
	uint32_t next() {
		x ^= x >> 6;
		x ^= x << 17;
		x ^= x >> 9;
		return uint32_t(x);
	}
	float operator()(void) {
		union {
			float f;
			uint32_t i;
		} u;
		u.i = (next() & fmask) | 0x3f800000;
		return u.f - 1.f;
	}
};

const Sphere spheres_glob[] = {
	// Scene: radius, position, emission, color, material
	Sphere(1e4, Vec(1e4 + 1, 40.8, 81.6), Vec(), Vec(.75, .25, .25),
	DIFF), // Left
	Sphere(1e4, Vec(-1e4 + 99, 40.8, 81.6), Vec(), Vec(.25, .25, .75),
	DIFF),                                                     // Rght
	Sphere(1e4, Vec(50, 40.8, 1e4), Vec(), Vec(.75, .75, .75), DIFF), // Back
	Sphere(1e4, Vec(50, 40.8, -1e4 + 170), Vec(), Vec(), DIFF),       // Frnt
	Sphere(1e4, Vec(50, 1e4, 81.6), Vec(), Vec(.75, .75, .75), DIFF), // Botm
	Sphere(1e4, Vec(50, -1e4 + 81.6, 81.6), Vec(), Vec(.75, .75, .75),
	DIFF),                                                      // Top
	Sphere(16.5, Vec(27, 16.5, 47), Vec(), Vec(1, 1, 1) * .999, SPEC), // Mirr
	Sphere(16.5, Vec(73, 16.5, 78), Vec(), Vec(1, 1, 1) * .999, REFR), // Glas
	Sphere(600, Vec(50, 681.6 - .27, 81.6), Vec(12, 12, 12), Vec(),
	DIFF) // Lite
};
inline int toInt(float x) {
	return int(sycl::pow(clamp(x), 1 / 2.2f) * 255 + .5f);
}

template<typename T>
inline bool intersect(const Ray &r, float &t, int &id,
	T spheres) {
	float d, inf = t = 1e20f;
	for(int i = 9; i--;)
		if((d = spheres[i].intersect(r)) && d < t) {
			t = d;
			id = i;
		}
	return t < inf;
}

template<typename T>
Vec radiance(const Ray &r_, int depth_, T spheres,
	RNG &rng) {
	float t;
	int id = 0;
	Ray r = r_;
	int depth = depth_;
	Vec cl(0, 0, 0); // accumulated color
	Vec cf(1, 1, 1); // accumulated reflectance
	while(1) {
		if(!intersect(r, t, id, spheres))
			return cl;                     // if miss, return black
		const Sphere &obj = spheres[id]; // the hit object
		Vec x = r.o + r.d * t, n = (x - obj.p).norm(),
			nl = n.dot(r.d) < 0 ? n : n * -1, f = obj.c;
		float p = f.x > f.y && f.x > f.z ? f.x : f.y > f.z ? f.y : f.z; // max refl
		cl = cl + cf.mult(obj.e);
		if(++depth > 5) {
			if(rng() < p) {
				f = f * (1 / p);
			}
			else {
				return cl;
			}
		} // R.R.
		cf = cf.mult(f);
		if(obj.refl == DIFF) { // Ideal DIFFUSE reflection
			float r1 = 2 * M_PI * rng(), r2 = rng(), r2s = sycl::sqrt(r2);
			Vec w = nl,
				u = ((sycl::fabs(w.x) > .1 ? Vec(0, 1) : Vec(1)) % w).norm(),
				v = w % u;
			Vec d = (u * sycl::cos(r1) * r2s + v * sycl::sin(r1) * r2s +
				w * sycl::sqrt(1 - r2)).norm();
			r = Ray(x, d);
			continue;
		}
		else if(obj.refl == SPEC) { // Ideal SPECULAR reflection
			r = Ray(x, r.d - n * 2 * n.dot(r.d));
			continue;
		}
		Ray reflRay(x, r.d - n * 2 * n.dot(r.d)); // Ideal dielectric REFRACTION
		bool into = n.dot(nl) > 0;                // Ray from outside going in?
		float nc = 1, nt = 1.5, nnt = into ? nc / nt : nt / nc, ddn = r.d.dot(nl),
			cos2t;
		if((cos2t = 1 - nnt * nnt * (1 - ddn * ddn)) <
			0) { // Total internal reflection
			r = reflRay;
			continue;
		}
		Vec tdir =
			(r.d * nnt -
			n * ((into ? 1 : -1) * (ddn * nnt + sycl::sqrt(cos2t)))).norm();
		float a = nt - nc, b = nt + nc, R0 = a * a / (b * b),
			c = 1 - (into ? -ddn : tdir.dot(n));
		float Re = R0 + (1 - R0) * c * c * c * c * c, Tr = 1 - Re,
			P = .25 + .5 * Re, RP = Re / P, TP = Tr / (1 - P);
		if(rng() < P) {
			cf = cf * RP;
			r = reflRay;
		}
		else {
			cf = cf * TP;
			r = Ray(x, tdir);
		}
		continue;
	}
}

// creating SYCL accessor types used by the command group.
typedef accessor<Vec, 1, access::mode::write, access::target::global_buffer>
vec_access;
typedef accessor<Sphere, 1, access::mode::read, access::target::constant_buffer>
spheres_access;

class kernel_r {
public:
	vec_access c;            // framebuffer
	spheres_access spheres_; // spheres
	int w, h, samps;

	Ray cam;
	Vec cx, cy, r;

	void operator()(item<2> item_) {
		const Sphere *spheres = &spheres_[0];
		const int x = item_[0];
		const int y = item_[1];
		RNG rng(1 + (y * w) + x); // initialise our own rng with rand() seed
		for(int sy = 0, i = (h - y - 1) * w + x; sy < 2; sy++) // 2x2 subpixel rows
			for(int sx = 0; sx < 2; sx++, r = Vec()) {           // 2x2 subpixel cols
				for(int s = 0; s < samps; s++) {
					float r1 = 2 * rng(), dx = r1 < 1 ? sycl::sqrt(r1) - 1
						: 1 - sycl::sqrt(2 - r1);
					float r2 = 2 * rng(), dy = r2 < 1 ? sycl::sqrt(r2) - 1
						: 1 - sycl::sqrt(2 - r2);
					Vec d = cx * (((sx + .5 + dx) / 2 + x) / w - .5) +
						cy * (((sy + .5 + dy) / 2 + y) / h - .5) + cam.d;
					r = r +
						radiance(Ray(cam.o + d * 140, d.norm()), 0, spheres, rng) *
						(1. / samps);
				} // Camera rays are pushed ^^^^^ forward to start in interior
				c[i] = c[i] + Vec(clamp(r.x), clamp(r.y), clamp(r.z)) * .25;
			}
	}
};

void compute_sycl(void* dev, int w, int h, int samps, Ray cam_, Vec cx_, Vec cy_, Vec r_, Vec* c) {
	queue q(*(device*)dev);
	{
		// data is wrapped in SYCL buffers.
		buffer<Vec, 1> color_buffer(c, range<1>(w * h));
		buffer<Sphere, 1> spheres_buffer(&spheres_glob[0], range<1>(9));
		auto cg = [&](handler &ch) {
			kernel_r ray_ = {
				// enabling access of the data on the device for SYCL.
				color_buffer.get_access<access::mode::write>(ch),
				spheres_buffer.get_access<access::mode::read,
				access::target::constant_buffer>(ch),
				w, h, samps,
				cam_,
				cx_, cy_, r_
			};
			nd_range<2> ndr(range<2>(w, h), range<2>(8, 8));
			ch.parallel_for(ndr, ray_);
		};
		// submitting the command group to the SYCL command queue for execution.
		q.submit(cg);
	}
}

int main(int argc, char** argv) {
	using namespace std;
	vector<testInfo> tests;

	// TODO: compute_sycl_gtx does not work with ComputeCpp yet
	getDevices(tests, { compute_sycl, /*compute_sycl_gtx*/ });

	return mainTester(argc, argv, tests);
}
