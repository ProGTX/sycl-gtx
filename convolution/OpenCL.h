#pragma once

#include <CL/cl.h>

#include <string>


struct OpenCL {
	using string = std::string;

	static string read(string filename);

	static const char* getErrorString(cl_int error);
	static void checkError(cl_int error);

	static void global(
		int numInvocations, cl_device_id dev,
		int width, int height,
		int filter_size, int filter_data_size,
		const float* input, float* output, const float* filter
	);

	static void local(
		int numInvocations, cl_device_id dev,
		int width, int height,
		int filter_size, int filter_data_size,
		const float* input, float* output, const float* filter
	);

private:
	static void common(
		int numInvocations, cl_device_id dev, string filename,
		int width, int height,
		int filter_size, int filter_data_size,
		const float* input, float* output, const float* filter,
		bool isLocal
	);
};
