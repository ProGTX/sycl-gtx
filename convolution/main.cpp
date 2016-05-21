#include "CPU.h"
#include "Filter.h"
#include "OpenCL.h"
#include "PPM.h"
#include "Sycl.h"
#include "Timer.h"

#include <algorithm>
#include <map>
#include <sycl.hpp>


int main() {
	using function_ptr_t = void(*)(
		int, cl_device_id, int, int, int, int,
		const float*, float*, const float*
	);

	using namespace std;

#define TEST_PAIR(name) { #name, name }
	map<string, function_ptr_t> tests = {
		TEST_PAIR(CPU::basic),
		TEST_PAIR(CPU::openmp),
		TEST_PAIR(CPU::threaded),
		TEST_PAIR(OpenCL::global),
		TEST_PAIR(OpenCL::local),
		TEST_PAIR(Sycl::local),
		TEST_PAIR(Sycl::local),
	};
#undef TEST_PAIR

	auto selector = cl::sycl::cpu_selector();
	auto device = cl::sycl::device(selector);
	auto image = PPM("test.ppm");

	static const int filterSize = 3;
	auto filter = Filter::generate(filterSize);

	auto input = image.toInternal();
	auto output = decltype(input)(input.size());
	Timer globalTimer;
	Timer localTimer;
	const float toSeconds = 1e-6f;

	cout << "Starting execution" << endl;

	for(auto&& test : tests) {
		cout << "Running test " << test.first << endl;
		
		int numIterations = 1;
		globalTimer.reset();
		localTimer.reset();
		test.second(
			numIterations, device.get(), image.width, image.height,
			filterSize, static_cast<int>(filter.size()),
			input.data(), output.data(), filter.data()
		);
		cout << "Kernel time: " <<
			(localTimer.elapsed() * toSeconds / numIterations) << endl;
		cout << "Total test time: " << (globalTimer.elapsed() * toSeconds) << endl;

		//auto outImage = PPM(output, image.width, image.height);
		//string filename = test.first;
		//replace(filename.begin(), filename.end(), ':', '_');
		//outImage.store(filename + ".ppm");
	}

	cout << "Finished benchmark" << endl;

	return 0;
}
