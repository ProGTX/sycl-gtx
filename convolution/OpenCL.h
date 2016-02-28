#pragma once

#include <CL/cl.h>

#include <string>


struct OpenCL {
	using string = std::string;

	static string read(string filename);

	static const char* getErrorString(cl_int error);
	static void checkError(cl_int error);

	static void global(
		int numInvocations, cl_device_id dev, string filename,
		int width, int height,
		int dataSize, int filterSize, int filterDataSize,
		const float* input, float* output, const float* filter
	);
};
