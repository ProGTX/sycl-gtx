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
auto duration = [](time_point before) {
	static const float to_seconds = 1e-6f;
	return std::chrono::duration_cast<std::chrono::microseconds>(now() - before).count() * to_seconds;
};

struct testInfo {
	using function_ptr = void(*)(int, int, int, Ray&, Vec&, Vec&, Vec, Vec*);
	string name;
	function_ptr test;
	float lastTime = 0;

	static decltype(now()) startTime;
	static float totalTime;

	testInfo(string name, function_ptr test)
		: name(name), test(test) {}
};
decltype(now()) testInfo::startTime = now();
float testInfo::totalTime = 0;

static std::vector<const testInfo> tests = {
	testInfo("org", compute_org),
	testInfo("openmp", compute_org_openmp),
	testInfo("sycl_cpu", compute_sycl_gtx_cpu),
	testInfo("sycl_gpu", compute_sycl_gtx_gpu),
	testInfo("org_single", compute_org_sp),
	testInfo("openmp_single", compute_org_sp_openmp),
};
static Ray cam(Vec(50, 52, 295.6), Vec(0, -0.042612, -1).norm()); // cam pos, dir

bool tester(int w, int h, int samples, Vec& cx, Vec& cy, int iterations, int from, int to) {
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
			time = (duration(start) / (float)iterations);
		}
		catch(cl::sycl::exception& e) {
			cerr << "SYCL error while testing: " << e.what() << endl;
			continue;
		}
		catch(exception& e) {
			cerr << "error while testing: " << e.what() << endl;
			continue;
		}
		cout << "time: " << time << endl;
		//to_file(w, h, vectors.data(), string("image_") + t.name + ".ppm");

		t.lastTime = time;

		testInfo::totalTime = duration(testInfo::startTime);
		if(testInfo::totalTime > 600) {
			cout << "exceeded 10 minute limit, stopping" << endl;
			return false;
		}
	}
	
	return true;
}

template <class T>
void printInfo(string description, const T& data, int offset = 0) {
	string indent;
	for(int i = 0; i < offset; ++i) {
		indent += '\t';
	}
	std::cout << indent << description << ": " << data << std::endl;
}

void displayPlaftormInfo() {
	using namespace std;

	try {
		using namespace cl::sycl;

		auto platforms = platform::get_platforms();

		int pNum = 0;
		for(auto& p : platforms) {
			cout << "- OpenCL platform " << pNum << ':' << endl;
			++pNum;

			printInfo("name", p.get_info<info::platform::name>(), 1);
			printInfo("vendor", p.get_info<info::platform::vendor>(), 1);
			printInfo("version", p.get_info<info::platform::version>(), 1);
			printInfo("profile", p.get_info<info::platform::profile>(), 1);
			printInfo("extensions", p.get_info<info::platform::extensions>(), 1);

			auto devices = p.get_devices();
			int dNum = 0;

			for(auto& d : devices) {
				cout << "\t-- OpenCL device " << dNum << ':' << endl;

				printInfo("name", d.get_info<info::device::name>(), 2);
				printInfo("device_type", (cl_device_type)d.get_info<info::device::device_type>(), 2);
				printInfo("vendor", d.get_info<info::device::vendor>(), 2);
				printInfo("device_version", d.get_info<info::device::device_version>(), 2);
				printInfo("driver_version", d.get_info<info::device::driver_version>(), 2);
				printInfo("opencl_version", d.get_info<info::device::opencl_version>(), 2);
				printInfo("single_fp_config", d.get_info<info::device::single_fp_config>(), 2);
				printInfo("double_fp_config", d.get_info<info::device::double_fp_config>(), 2);
				printInfo("profile", d.get_info<info::device::profile>(), 2);
				printInfo("error_correction_support", d.get_info<info::device::error_correction_support>(), 2);
				printInfo("host_unified_memory", d.get_info<info::device::host_unified_memory>(), 2);
				printInfo("max_clock_frequency", d.get_info<info::device::max_clock_frequency>(), 2);
				printInfo("max_compute_units", d.get_info<info::device::max_compute_units>(), 2);
				printInfo("max_work_item_dimensions", d.get_info<info::device::max_work_item_dimensions>(), 2);
				printInfo("max_work_group_size", d.get_info<info::device::max_work_group_size>(), 2);

				printInfo("address_bits", d.get_info<info::device::address_bits>(), 2);
				printInfo("max_mem_alloc_size", d.get_info<info::device::max_mem_alloc_size>(), 2);
				printInfo("global_mem_cache_line_size", d.get_info<info::device::global_mem_cache_line_size>(), 2);
				printInfo("global_mem_cache_size", d.get_info<info::device::global_mem_cache_size>(), 2);
				printInfo("global_mem_size", d.get_info<info::device::global_mem_size>(), 2);
				printInfo("max_constant_buffer_size", d.get_info<info::device::max_constant_buffer_size>(), 2);
				printInfo("max_constant_args", d.get_info<info::device::max_constant_args>(), 2);
				printInfo("local_mem_size", d.get_info<info::device::local_mem_size>(), 2);
				printInfo("extensions", d.get_info<info::device::extensions>(), 2);

				++dNum;
			}
		}
	}
	catch(cl::sycl::exception& e) {
		// TODO
		cout << "OpenCL not available: " << e.what() << endl;
	}
}

int main(int argc, char *argv[]) {
	using namespace std;

	cout << "smallpt SYCL tester" << endl;
	displayPlaftormInfo();

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
		iterations = 1;
		bool canContinue;

		for(samples = 10; samples < 10000; samples *= 2) {
			canContinue = tester(w, h, samples, cx, cy, iterations, 3, 4);
			if(!canContinue) {
				break;
			}
		}

		auto time = duration(testInfo::startTime);
		cout << "total test suite duration: " << time << endl;
	}

	//cout << "Press any key to exit" << endl;
	//cin.get();

	return 0;
}
