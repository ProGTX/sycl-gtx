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
		stream << line;
	}

	return stream.str();
}

void OpenCL::global(
	int numInvocations, cl_device_id dev, string filename,
	int dataSize, int filterSize,
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
	if(program.build({ device }) != CL_SUCCESS) {
		auto log = program.getBuildInfo<CL_PROGRAM_BUILD_LOG>(device, &error);
		cerr << "Error building:" << endl << log << endl;
		throw runtime_error(string("Error building: ") + to_string(error));
	}

	cl::Buffer bufInput(context, CL_MEM_READ_ONLY, sizeof(float) * dataSize);
	cl::Buffer bufOutput(context, CL_MEM_WRITE_ONLY, sizeof(float) * dataSize);
	cl::Buffer bufFilter(context, CL_MEM_READ_ONLY, sizeof(float) * filterSize);

	cl::CommandQueue queue(context, device);

	queue.enqueueWriteBuffer(bufFilter, CL_FALSE, 0, sizeof(float) * filterSize, filter);
	queue.enqueueWriteBuffer(bufInput, CL_TRUE, 0, sizeof(float) * dataSize, input);

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
