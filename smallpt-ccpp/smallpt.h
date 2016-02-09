/******************************************************************************
*
* Modified version for SYCL of Kevin Beason smallpt
* http://www.kevinbeason.com/smallpt/
*
******************************************************************************/

#include <SYCL/sycl.hpp>

using namespace cl::sycl;

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

struct Vec {     // Usage: time ./smallpt 5000 && xv image.ppm
	float x, y, z; // position, also color (r,g,b)
	Vec(float x_ = 0, float y_ = 0, float z_ = 0) : x(x_), y(y_), z(z_) {}
	Vec operator+(const Vec &b) const { return Vec(x + b.x, y + b.y, z + b.z); }
	Vec operator-(const Vec &b) const { return Vec(x - b.x, y - b.y, z - b.z); }
	Vec operator*(float b) const { return Vec(x * b, y * b, z * b); }
	Vec mult(const Vec &b) const { return Vec(x * b.x, y * b.y, z * b.z); }
	Vec &norm() {
		return *this = *this * (1 / sqrt(x * x + y * y + z * z));
	}
	float dot(const Vec &b) const {
		return x * b.x + y * b.y + z * b.z;
	} // cross:
	Vec operator%(Vec &b) {
		return Vec(y * b.z - z * b.y, z * b.x - x * b.z, x * b.y - y * b.x);
	}
};
struct Ray {
	Vec o, d;
	Ray(Vec o_, Vec d_) : o(o_), d(d_) {}
};
enum Refl_t { DIFF, SPEC, REFR }; // material types, used in radiance()
struct Sphere {
	float rad;   // radius
	Vec p, e, c; // position, emission, color
	Refl_t refl; // reflection type (DIFFuse, SPECular, REFRactive)
	Sphere(float rad_, Vec p_, Vec e_, Vec c_, Refl_t refl_)
		: rad(rad_), p(p_), e(e_), c(c_), refl(refl_) {}
	inline float intersect(const Ray &r) const { // returns distance, 0 if nohit
		Vec op = p - r.o; // Solve t^2*d.d + 2*t*(o-p).d + (o-p).(o-p)-R^2 = 0
		float t, eps = 1.5e-2f, b = op.dot(r.d),
			det = b * b - op.dot(op) + rad * rad;
		if(det < 0)
			return 0;
		else
			det = sqrt(det);
		return (t = b - det) > eps ? t : ((t = b + det) > eps ? t : 0);
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
inline float clamp(float x) { return x < 0 ? 0 : x > 1 ? 1 : x; }
inline int toInt(float x) {
	return int(pow(clamp(x), 1 / 2.2f) * 255 + .5f);
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
			float r1 = 2 * M_PI * rng(), r2 = rng(), r2s = sqrt(r2);
			Vec w = nl,
				u = ((fabs(w.x) > .1 ? Vec(0, 1) : Vec(1)) % w).norm(),
				v = w % u;
			Vec d = (u * cos(r1) * r2s + v * sin(r1) * r2s +
				w * sqrt(1 - r2)).norm();
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
			n * ((into ? 1 : -1) * (ddn * nnt + sqrt(cos2t)))).norm();
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

	void operator()(item<2> item_) {
		const Sphere *spheres = &spheres_[0];
		const int x = item_[0];
		const int y = item_[1];
		Vec r;
		Ray cam(Vec(50, 52, 295.6), Vec(0, -0.042612, -1).norm()); // cam pos, dir
		Vec cx = Vec(w * .5135 / h), cy = (cx % cam.d).norm() * .5135;
		RNG rng(1 + (y * w) + x); // initialise our own rng with rand() seed
		for(int sy = 0, i = (h - y - 1) * w + x; sy < 2; sy++) // 2x2 subpixel rows
			for(int sx = 0; sx < 2; sx++, r = Vec()) {           // 2x2 subpixel cols
				for(int s = 0; s < samps; s++) {
					float r1 = 2 * rng(), dx = r1 < 1 ? sqrt(r1) - 1
						: 1 - sqrt(2 - r1);
					float r2 = 2 * rng(), dy = r2 < 1 ? sqrt(r2) - 1
						: 1 - sqrt(2 - r2);
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

int main(int argc, char *argv[]) {
	gpu_selector s_;
	queue q_(s_);
	int w = 1024, h = 768, samps = argc == 2 ? atoi(argv[1]) / 4 : 1; // # samples
	Vec *c = new Vec[w * h];
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
				w, h, samps };
			nd_range<2> ndr(range<2>(w, h), range<2>(8, 8));
			ch.parallel_for(ndr, ray_);
		};
		// submitting the command group to the SYCL command queue for execution.
		q_.submit(cg);
	}
	FILE *f = fopen("image.ppm", "w"); // Write image to PPM file.
	fprintf(f, "P3\n%d %d\n%d\n", w, h, 255);
	for(int i = 0; i < w * h; i++)
		fprintf(f, "%d %d %d ", toInt(c[i].x), toInt(c[i].y), toInt(c[i].z));
}
