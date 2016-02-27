#pragma once

#include <CL/cl.hpp>

#include <string>


struct OpenCL {
	using string = std::string;

	static string read(string filename);

	static void global(
		int numInvocations, cl_device_id dev, string filename,
		int dataSize, int filterSize,
		const float* input, float* output, const float* filter
	);
};
