#include "classes.h"
#include "msvc.h"

#include <chrono>
#include <iostream>
#include <string>
#include <vector>

extern void compute_org(int w, int h, int samps, Ray& cam, Vec& cx, Vec& cy, Vec& r, Vec* c);
extern void compute_sycl_gtx(int w, int h, int samps, Ray& cam, Vec& cx, Vec& cy, Vec& r, Vec* c);

inline double clamp(double x) {
	return x < 0 ? 0 : x>1 ? 1 : x;
}
inline int toInt(double x) {
	return int(pow(clamp(x), 1 / 2.2) * 255 + .5);
}

void to_file(int w, int h, Vec* c, std::string filename) {
	FILE* f = fopen(filename.c_str(), "w");         // Write image to PPM file.
	fprintf(f, "P3\n%d %d\n%d\n", w, h, 255);
	for(int i = 0; i < w*h; i++) {
		fprintf(f, "%d %d %d\n", toInt(c[i].x), toInt(c[i].y), toInt(c[i].z));
	}
}

using time_point = std::chrono::high_resolution_clock::time_point;

auto now = []() {
	return std::chrono::high_resolution_clock::now();
};
auto duration = [](time_point before, time_point after) {
	return std::chrono::duration_cast<std::chrono::microseconds>(after - before).count();
};

int main(int argc, char *argv[]) {
	int w = 1024, h = 768, samps = argc == 2 ? atoi(argv[1]) / 4 : 1; // # samples
	Ray cam(Vec(50, 52, 295.6), Vec(0, -0.042612, -1).norm()); // cam pos, dir
	Vec cx = Vec(w*.5135 / h), cy = (cx%cam.d).norm()*.5135, r, *c = new Vec[w*h];

	std::vector<Vec> vectors_org(w*h, 0);
	auto vector_sycl_gtx(vectors_org);

	int iterations = 1;

	ns_erand::reset();
	auto start = now();
	for(int i = 0; i < iterations; ++i) {
		compute_org(w, h, samps, cam, cx, cy, r, vectors_org.data());
	}
	std::cout << "\n" << (duration(start, now()) / (float)iterations) << std::endl;
	to_file(w, h, vectors_org.data(), "image_org.ppm");

	ns_erand::reset();
	start = now();
	for(int i = 0; i < iterations; ++i) {
		compute_sycl_gtx(w, h, samps, cam, cx, cy, r, vector_sycl_gtx.data());
	}
	std::cout << "\n" << (duration(start, now()) / (float)iterations) << std::endl;
	to_file(w, h, vector_sycl_gtx.data(), "image_sycl_gtx.ppm");

	std::cin.get();

	return 0;
}
