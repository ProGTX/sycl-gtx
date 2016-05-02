#pragma once

#include <CL/cl.h>
#include <string>


struct Sycl {
	using string = std::string;

	static void local(
		int num_invocations, cl_device_id dev,
		int width, int height,
		int filter_size, int filter_data_size,
		const float* input, float* output, const float* filter
	);

	static void global(
		int num_invocations, cl_device_id dev,
		int width, int height,
		int filter_size, int filter_data_size,
		const float* input, float* output, const float* filter
	);
};
