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

// http://stackoverflow.com/a/24336429/793006
const char* OpenCL::getErrorString(cl_int error) {
	switch(error) {
		// run-time and JIT compiler errors
		case 0: return "CL_SUCCESS";
		case -1: return "CL_DEVICE_NOT_FOUND";
		case -2: return "CL_DEVICE_NOT_AVAILABLE";
		case -3: return "CL_COMPILER_NOT_AVAILABLE";
		case -4: return "CL_MEM_OBJECT_ALLOCATION_FAILURE";
		case -5: return "CL_OUT_OF_RESOURCES";
		case -6: return "CL_OUT_OF_HOST_MEMORY";
		case -7: return "CL_PROFILING_INFO_NOT_AVAILABLE";
		case -8: return "CL_MEM_COPY_OVERLAP";
		case -9: return "CL_IMAGE_FORMAT_MISMATCH";
		case -10: return "CL_IMAGE_FORMAT_NOT_SUPPORTED";
		case -11: return "CL_BUILD_PROGRAM_FAILURE";
		case -12: return "CL_MAP_FAILURE";
		case -13: return "CL_MISALIGNED_SUB_BUFFER_OFFSET";
		case -14: return "CL_EXEC_STATUS_ERROR_FOR_EVENTS_IN_WAIT_LIST";
		case -15: return "CL_COMPILE_PROGRAM_FAILURE";
		case -16: return "CL_LINKER_NOT_AVAILABLE";
		case -17: return "CL_LINK_PROGRAM_FAILURE";
		case -18: return "CL_DEVICE_PARTITION_FAILED";
		case -19: return "CL_KERNEL_ARG_INFO_NOT_AVAILABLE";

			// compile-time errors
		case -30: return "CL_INVALID_VALUE";
		case -31: return "CL_INVALID_DEVICE_TYPE";
		case -32: return "CL_INVALID_PLATFORM";
		case -33: return "CL_INVALID_DEVICE";
		case -34: return "CL_INVALID_CONTEXT";
		case -35: return "CL_INVALID_QUEUE_PROPERTIES";
		case -36: return "CL_INVALID_COMMAND_QUEUE";
		case -37: return "CL_INVALID_HOST_PTR";
		case -38: return "CL_INVALID_MEM_OBJECT";
		case -39: return "CL_INVALID_IMAGE_FORMAT_DESCRIPTOR";
		case -40: return "CL_INVALID_IMAGE_SIZE";
		case -41: return "CL_INVALID_SAMPLER";
		case -42: return "CL_INVALID_BINARY";
		case -43: return "CL_INVALID_BUILD_OPTIONS";
		case -44: return "CL_INVALID_PROGRAM";
		case -45: return "CL_INVALID_PROGRAM_EXECUTABLE";
		case -46: return "CL_INVALID_KERNEL_NAME";
		case -47: return "CL_INVALID_KERNEL_DEFINITION";
		case -48: return "CL_INVALID_KERNEL";
		case -49: return "CL_INVALID_ARG_INDEX";
		case -50: return "CL_INVALID_ARG_VALUE";
		case -51: return "CL_INVALID_ARG_SIZE";
		case -52: return "CL_INVALID_KERNEL_ARGS";
		case -53: return "CL_INVALID_WORK_DIMENSION";
		case -54: return "CL_INVALID_WORK_GROUP_SIZE";
		case -55: return "CL_INVALID_WORK_ITEM_SIZE";
		case -56: return "CL_INVALID_GLOBAL_OFFSET";
		case -57: return "CL_INVALID_EVENT_WAIT_LIST";
		case -58: return "CL_INVALID_EVENT";
		case -59: return "CL_INVALID_OPERATION";
		case -60: return "CL_INVALID_GL_OBJECT";
		case -61: return "CL_INVALID_BUFFER_SIZE";
		case -62: return "CL_INVALID_MIP_LEVEL";
		case -63: return "CL_INVALID_GLOBAL_WORK_SIZE";
		case -64: return "CL_INVALID_PROPERTY";
		case -65: return "CL_INVALID_IMAGE_DESCRIPTOR";
		case -66: return "CL_INVALID_COMPILER_OPTIONS";
		case -67: return "CL_INVALID_LINKER_OPTIONS";
		case -68: return "CL_INVALID_DEVICE_PARTITION_COUNT";

			// extension errors
		case -1000: return "CL_INVALID_GL_SHAREGROUP_REFERENCE_KHR";
		case -1001: return "CL_PLATFORM_NOT_FOUND_KHR";
		case -1002: return "CL_INVALID_D3D10_DEVICE_KHR";
		case -1003: return "CL_INVALID_D3D10_RESOURCE_KHR";
		case -1004: return "CL_D3D10_RESOURCE_ALREADY_ACQUIRED_KHR";
		case -1005: return "CL_D3D10_RESOURCE_NOT_ACQUIRED_KHR";
		default: return "Unknown OpenCL error";
	}
}

void OpenCL::checkError(cl_int error) {
	using namespace std;
	if(error != CL_SUCCESS) {
		stringstream message;
		message << "OpenCL error (" << error << "): " << getErrorString(error);
		cerr << message.str() << endl;
		throw runtime_error(message.str());
	}
}

void OpenCL::global(
	int numInvocations, cl_device_id dev, string filename, string compileOptions,
	int width, int height,
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

	cl::Program program(context, sources, &error);
	checkError(error);
	if(program.build({ device }, compileOptions.c_str()) != CL_SUCCESS) {
		auto log = program.getBuildInfo<CL_PROGRAM_BUILD_LOG>(device, &error);
		cerr << "Error building:" << endl << log << endl;
		checkError(error);
	}

	cl::Buffer bufInput(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, sizeof(float) * dataSize, (void*)input, &error);
	checkError(error);
	cl::Buffer bufOutput(context, CL_MEM_WRITE_ONLY, sizeof(float) * dataSize, &error);
	checkError(error);
	cl::Buffer bufFilter(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, sizeof(float) * filterDataSize, (void*)filter, &error);
	checkError(error);

	cl::CommandQueue queue(context, device, 0, &error);
	checkError(error);

	cl::Kernel convolution = cl::Kernel(program, "convolute", &error);
	checkError(error);
	error = convolution.setArg(0, bufInput());
	checkError(error);
	error = convolution.setArg(1, bufOutput());
	checkError(error);
	error = convolution.setArg(2, bufFilter());
	checkError(error);
	for(int i = 0; i < numInvocations; ++i) {
		error = queue.enqueueNDRangeKernel(convolution, cl::NullRange, cl::NDRange(width, height));
		checkError(error);
		error = queue.finish();
		checkError(error);
	}

	error = queue.enqueueReadBuffer(bufOutput, CL_TRUE, 0, sizeof(float) * dataSize, output);
	checkError(error);
}
