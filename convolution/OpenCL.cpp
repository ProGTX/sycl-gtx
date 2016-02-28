#include "OpenCL.h"

#include <fstream>
#include <iostream>
#include <sstream>


using namespace std;

string OpenCL::read(string filename) {
	using namespace std;

	auto file = ifstream(filename);
	ostringstream stream;
	string line;

	while(file.good()) {
		getline(file, line);
		stream << line << endl;
	}

	return stream.str();
}

void OpenCL::global(
	int numInvocations, cl_device_id dev, string filename, string compileOptions,
	int dataSize, int filterDataSize,
	const float* input, float* output, const float* filter
) {
	using namespace std;
	cl_int error;

	cl::Device device(dev);
	cl::Context context({ device });

	cl::Program::Sources sources;
	string kernel_code = read(filename);
	sources.push_back({ kernel_code.c_str(), kernel_code.length() });

	cl::Program program(context, sources);
	if(program.build({ device }, compileOptions.c_str()) != CL_SUCCESS) {
		auto log = program.getBuildInfo<CL_PROGRAM_BUILD_LOG>(device, &error);
		cerr << "Error building:" << endl << log << endl;
		throw runtime_error(string("Error building: ") + to_string(error));
	}

	cl::Buffer bufInput(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, sizeof(float) * dataSize, (void*)input, &error);
	cl::Buffer bufOutput(context, CL_MEM_WRITE_ONLY, sizeof(float) * dataSize, &error);
	cl::Buffer bufFilter(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, sizeof(float) * filterDataSize, (void*)filter, &error);

	cl::CommandQueue queue(context, device);

	cl::Kernel convolution = cl::Kernel(program, "convolute");
	convolution.setArg(0, input);
	convolution.setArg(1, output);
	convolution.setArg(2, filter);
	for(int i = 0; i < numInvocations; ++i) {
		queue.enqueueNDRangeKernel(convolution, cl::NullRange, cl::NDRange(dataSize), cl::NullRange);
		queue.finish();
	}

	queue.enqueueReadBuffer(bufOutput, CL_TRUE, 0, sizeof(float) * dataSize, output);
}
