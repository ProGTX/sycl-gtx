#include "OpenCL.h"

#include <fstream>
#include <iostream>
#include <sstream>
#include <vector>


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
#ifdef _DEBUG
	using namespace std;
	if(error != CL_SUCCESS) {
		stringstream message;
		message << "OpenCL error (" << error << "): " << getErrorString(error);
		cerr << message.str() << endl;
		throw runtime_error(message.str());
	}
#endif
}

void OpenCL::global(
	int numInvocations, cl_device_id dev,
	int IMAGE_W, int IMAGE_H,
	int FILTER_SIZE, int filter_data_size,
	const float* input, float* output, const float* filter
) {
	common(
		numInvocations, dev, "global_opt.cl",
		IMAGE_W, IMAGE_H,
		FILTER_SIZE, filter_data_size,
		input, output, filter,
		false
	);
}

void OpenCL::local(
	int numInvocations, cl_device_id dev,
	int IMAGE_W, int IMAGE_H,
	int FILTER_SIZE, int filter_data_size,
	const float* input, float* output, const float* filter
) {
	common(
		numInvocations, dev, "local_opt.cl",
		IMAGE_W, IMAGE_H,
		FILTER_SIZE, filter_data_size,
		input, output, filter,
		true
	);
}

void OpenCL::common(
	int numInvocations, cl_device_id dev, string filename,
	int IMAGE_W, int IMAGE_H,
	int FILTER_SIZE, int filter_data_size,
	const float* input, float* output, const float* filter,
	bool isLocal
) {
	using namespace std;
	cl_int error;

	int dataSize = IMAGE_W * IMAGE_H * 4;
	auto context = clCreateContext(nullptr, 1, &dev, nullptr, nullptr, &error);

	string kernelCode = read(filename);

	auto codePtr = kernelCode.c_str();
	auto codeLength = kernelCode.length();
	auto program = clCreateProgramWithSource(context, 1, &codePtr, &codeLength, &error);
	checkError(error);

	char compileOptions[1024];
	sprintf(
		compileOptions,
		"-D IMAGE_W=%d -D IMAGE_H=%d -D FILTER_SIZE=%d "
		"-D HALF_FILTER_SIZE=%d -D TWICE_HALF_FILTER_SIZE=%d -D HALF_FILTER_SIZE_IMAGE_W=%d",
		IMAGE_W, IMAGE_H, FILTER_SIZE,
		FILTER_SIZE / 2, (FILTER_SIZE / 2) * 2, (FILTER_SIZE / 2) * IMAGE_W
	);

	error = clBuildProgram(program, 1, &dev, compileOptions, nullptr, nullptr);
	if(error != CL_SUCCESS) {
		size_t length;
		error = clGetProgramBuildInfo(program, dev, CL_PROGRAM_BUILD_LOG, 0, nullptr, &length);
		checkError(error);

		auto log = vector<char>(length);
		error = clGetProgramBuildInfo(program, dev, CL_PROGRAM_BUILD_LOG, length, log.data(), nullptr);
		cerr << "Error building:" << endl << string(log.data()) << endl;
		checkError(error);
	}

	auto bufInput = clCreateBuffer(
		context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, sizeof(float) * dataSize, (void*)input, &error
	);
	checkError(error);
	auto bufOutput = clCreateBuffer(
		context, CL_MEM_WRITE_ONLY, sizeof(float) * dataSize, nullptr, &error
	);
	checkError(error);
	auto bufFilter = clCreateBuffer(
		context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, sizeof(float) * filter_data_size, (void*)filter, &error
	);
	checkError(error);

	auto queue = clCreateCommandQueue(context, dev, 0, &error);
	checkError(error);

	auto kernel = clCreateKernel(program, "convolute", &error);
	checkError(error);

	error = clSetKernelArg(kernel, 0, sizeof(cl_mem), (void*)&bufInput);
	checkError(error);
	error = clSetKernelArg(kernel, 1, sizeof(cl_mem), (void*)&bufOutput);
	checkError(error);
	error = clSetKernelArg(kernel, 2, sizeof(cl_mem), (void*)&bufFilter);
	checkError(error);

	size_t localWorkSize[] = { 16, 16 };
	size_t globalWorkSize[] = { IMAGE_W, IMAGE_H };

	if(isLocal) {
		auto localMemSize =
			(localWorkSize[0] + 2 * (FILTER_SIZE / 2)) *
			(localWorkSize[1] + 2 * (FILTER_SIZE / 2));
		localMemSize *= 4 * sizeof(float);

		error = clSetKernelArg(kernel, 3, localMemSize, nullptr);
		checkError(error);
	}

	for(int i = 0; i < numInvocations; ++i) {
		error = clEnqueueNDRangeKernel(
			queue, kernel, 2,
			nullptr, globalWorkSize, (isLocal ? localWorkSize : nullptr),
			0, nullptr, nullptr
		);
		checkError(error);
		error = clFinish(queue);
		checkError(error);
	}

	error = clEnqueueReadBuffer(
		queue, bufOutput, CL_TRUE, 0, sizeof(float) * dataSize, output, 0, nullptr, nullptr
	);
	checkError(error);
}
