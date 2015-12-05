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

struct testInfo {
	using function_ptr = void(*)(int, int, int, Ray&, Vec&, Vec&, Vec, Vec*);
	string name;
	function_ptr test;
	float lastTime = 0;

	testInfo(string name, function_ptr test)
		: name(name), test(test) {}
};

static std::vector<const testInfo> tests = {
	testInfo("org", compute_org),
	testInfo("openmp", compute_org_openmp),
	testInfo("sycl_cpu", compute_sycl_gtx_cpu),
	testInfo("sycl_gpu", compute_sycl_gtx_gpu),
	testInfo("org_single", compute_org_sp),
	testInfo("openmp_single", compute_org_sp_openmp),
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
		auto& t = tests[ti];

		// Quality of Service
		// Prevent the user from waiting too long
		if(t.lastTime > 80) {
			continue;
		}

		cout << "Running test: " << t.name << endl;
		ns_erand::reset();
		try {
			auto start = now();
			for(int i = 0; i < iterations; ++i) {
				vectors = empty_vectors;
				t.test(w, h, samples, cam, cx, cy, r, vectors.data());
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
		//to_file(w, h, vectors.data(), string("image_") + t.name + ".ppm");

		t.lastTime = time;
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
		iterations = 1;

		for(samples = 10; samples < 10000; samples *= 2) {
			tester(w, h, samples, cx, cy, iterations, 0, 4);
		}

		auto time = duration(start, now()) * to_seconds;
		cout << "total test suite duration: " << time << endl;
	}

	//cout << "Press any key to exit" << endl;
	//cin.get();

	return 0;
}
