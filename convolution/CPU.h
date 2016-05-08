#pragma once

#include <CL/cl.h>

struct CPU {
private:
	static void innerLoop(
		cl_device_id dev,
		int IMAGE_W, int IMAGE_H,
		int filter_size, int filter_data_size,
		const float* input_, float* output_, const float* filter_,
		int y);

public:
	static void basic(
		int num_invocations, cl_device_id dev,
		int IMAGE_W, int IMAGE_H,
		int filter_size, int filter_data_size,
		const float* input_, float* output_, const float* filter_
	);

	static void openmp(
		int num_invocations, cl_device_id dev,
		int IMAGE_W, int IMAGE_H,
		int filter_size, int filter_data_size,
		const float* input_, float* output_, const float* filter_
	);

	static void threaded(
		int num_invocations, cl_device_id dev,
		int IMAGE_W, int IMAGE_H,
		int filter_size, int filter_data_size,
		const float* input_, float* output_, const float* filter_
	);
};
