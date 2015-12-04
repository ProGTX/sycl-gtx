#include "classes.h"
#include "msvc.h"

#include <sycl.hpp>

#include <chrono>
#include <iostream>
#include <string>
#include <vector>


using std::string;

extern void compute_org(int w, int h, int samps, Ray& cam, Vec& cx, Vec& cy, Vec r, Vec* c);
extern void compute_org_openmp(int w, int h, int samps, Ray& cam, Vec& cx, Vec& cy, Vec r, Vec* c);
extern void compute_org_sp(int w, int h, int samps, Ray& cam, Vec& cx, Vec& cy, Vec r, Vec* c);
extern void compute_org_sp_openmp(int w, int h, int samps, Ray& cam, Vec& cx, Vec& cy, Vec r, Vec* c);
extern void compute_sycl_gtx_cpu(int w, int h, int samps, Ray& cam, Vec& cx, Vec& cy, Vec r, Vec* c);
extern void compute_sycl_gtx_gpu(int w, int h, int samps, Ray& cam, Vec& cx, Vec& cy, Vec r, Vec* c);

inline double clamp(double x) {
	return x < 0 ? 0 : x>1 ? 1 : x;
}
inline int toInt(double x) {
	return int(pow(clamp(x), 1 / 2.2) * 255 + .5);
}

void to_file(int w, int h, Vec* c, string filename) {
	FILE* f = fopen(filename.c_str(), "w");         // Write image to PPM file.
	fprintf(f, "P3\n%d %d\n%d\n", w, h, 255);
	for(int i = 0; i < w*h; i++) {
		fprintf(f, "%d %d %d\n", toInt(c[i].x), toInt(c[i].y), toInt(c[i].z));
	}
	fclose(f);
}

using time_point = std::chrono::high_resolution_clock::time_point;

auto now = []() {
	return std::chrono::high_resolution_clock::now();
};
auto duration = [](time_point before, time_point after) {
	return std::chrono::duration_cast<std::chrono::microseconds>(after - before).count();
};

static std::vector<std::pair<string, void(*)(int, int, int, Ray&, Vec&, Vec&, Vec, Vec*)>> tests = {
	{ "org", compute_org },
	{ "openmp", compute_org_openmp },
	{ "sycl_cpu", compute_sycl_gtx_cpu },
	{ "sycl_gpu", compute_sycl_gtx_gpu },
	{ "org_single", compute_org_sp },
	{ "openmp_single", compute_org_sp_openmp },
};
static Ray cam(Vec(50, 52, 295.6), Vec(0, -0.042612, -1).norm()); // cam pos, dir
static const float to_seconds = 1e-6f;

void tester(int w, int h, int samples, Vec& cx, Vec& cy, int iterations, int from, int to) {
	using namespace std;

	cout << "samples per pixel: " << samples << endl;

	Vec r;
	vector<Vec> empty_vectors(w*h, 0);
	vector<Vec> vectors;
	float time;

	for(int ti = from; ti < to; ++ti) {
		auto&& t = tests[ti];
		cout << "Running test: " << t.first << endl;
		ns_erand::reset();
		try {
			auto start = now();
			for(int i = 0; i < iterations; ++i) {
				vectors = empty_vectors;
				t.second(w, h, samples, cam, cx, cy, r, vectors.data());
			}
			time = (duration(start, now()) / (float)iterations);
		}
		catch(cl::sycl::exception& e) {
			cerr << "SYCL error while testing: " << e.what() << endl;
			continue;
		}
		catch(exception& e) {
			cerr << "error while testing: " << e.what() << endl;
			continue;
		}
		time *= to_seconds;
		cout << "time: " << time << endl;
		to_file(w, h, vectors.data(), string("image_") + t.first + ".ppm");
	}
}

int main(int argc, char *argv[]) {
	using namespace std;

	bool fromCommand = argc == 2;
	int w = 1024;
	int h = 768;
	int samples = fromCommand ? atoi(argv[1]) / 4 : 10;
	int iterations = 1;
	Vec cx = Vec(w*.5135 / h);
	Vec cy = (cx%cam.d).norm()*.5135;

	if(fromCommand) {
		tester(w, h, samples, cx, cy, iterations, 0, 4);
	}
	else {
		// Test suite

		auto start = now();

		samples = 10;
		iterations = 1;
		tester(w, h, samples, cx, cy, iterations, 0, 1);
		iterations = 2;
		tester(w, h, samples, cx, cy, iterations, 1, 4);

		samples = 20;
		iterations = 1;
		tester(w, h, samples, cx, cy, iterations, 1, 2);
		iterations = 2;
		tester(w, h, samples, cx, cy, iterations, 2, 4);

		samples = 40;
		iterations = 1;
		tester(w, h, samples, cx, cy, iterations, 1, 4);

		samples = 80;
		iterations = 1;
		tester(w, h, samples, cx, cy, iterations, 1, 4);

		auto time = duration(start, now()) * to_seconds;
		cout << "total test suite duration: " << time << endl;
	}

	//cout << "Press any key to exit" << endl;
	//cin.get();

	return 0;
}
