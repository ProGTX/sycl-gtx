#include "CPU.h"

#include <functional>
#include <thread>
#include <vector>


namespace {

cl_float4& operator+=(cl_float4& left, const cl_float4& right) {
	for(int i = 0; i < 4; ++i) {
		left.s[i] += right.s[i];
	}
	return left;
}
cl_float4 operator*(const cl_float4& left, const cl_float4& right) {
	cl_float4 result;
	for(int i = 0; i < 4; ++i) {
		result.s[i] = left.s[i] * right.s[i];
	}
	return result;
}

}

void CPU::innerLoop(
	cl_device_id dev,
	int IMAGE_W, int IMAGE_H,
	int filter_size, int filter_data_size,
	const float* input_, float* output_, const float* filter_,
	int y
	) {
	const int HALF_FILTER_SIZE = filter_size / 2;
	auto input = reinterpret_cast<const cl_float4*>(input_);
	auto output = reinterpret_cast<cl_float4*>(output_);
	auto filter = reinterpret_cast<const cl_float4*>(filter_);
	cl_int3 gid = { 0, y, 0 };

	for(gid.x = 0; gid.x < IMAGE_W; ++gid.x) {
		if(
			gid.x < HALF_FILTER_SIZE ||
			gid.x > IMAGE_W - HALF_FILTER_SIZE - 1 ||
			gid.y < HALF_FILTER_SIZE ||
			gid.y > IMAGE_H - HALF_FILTER_SIZE - 1
			) {
			// Not performed at the edges, but not much of a problem
			continue;
		}

		gid.z = gid.y * IMAGE_W + gid.x;

		// Convolution
		int filterPos = 0;
		cl_float4 sum = { 0, 0, 0, 0 };
		for(int r = -HALF_FILTER_SIZE; r <= HALF_FILTER_SIZE; ++r) {
			int inputPos = gid.z + IMAGE_W * r;
			for(int c = -HALF_FILTER_SIZE; c <= HALF_FILTER_SIZE; ++c) {
				sum += input[inputPos] * filter[filterPos];
				filterPos += 1;
				inputPos += 1;
			}
		}
		output[gid.z] = sum;
	}
}

void CPU::basic(
	int num_invocations, cl_device_id dev,
	int IMAGE_W, int IMAGE_H,
	int filter_size, int filter_data_size,
	const float* input_, float* output_, const float* filter_
) {
	for(int i = 0; i < num_invocations; ++i) {
		for(int y = 0; y < IMAGE_H; ++y) {
			innerLoop(
				dev,
				IMAGE_W, IMAGE_H,
				filter_size, filter_data_size,
				input_, output_, filter_,
				y);
		}
	}
}

void CPU::openmp(
	int num_invocations, cl_device_id dev,
	int IMAGE_W, int IMAGE_H,
	int filter_size, int filter_data_size,
	const float* input_, float* output_, const float* filter_
) {
	for(int i = 0; i < num_invocations; ++i) {
		#pragma omp parallel for schedule(static)
		for(int y = 0; y < IMAGE_H; ++y) {
			innerLoop(
				dev,
				IMAGE_W, IMAGE_H,
				filter_size, filter_data_size,
				input_, output_, filter_,
				y);
		}
	}
}

void CPU::threaded(
	int num_invocations, cl_device_id dev,
	int IMAGE_W, int IMAGE_H,
	int filter_size, int filter_data_size,
	const float* input_, float* output_, const float* filter_
) {
	using namespace std;

	vector<thread> threads;
	threads.reserve(IMAGE_H);

	for(int i = 0; i < num_invocations; ++i) {
		threads.clear();
		for(int y = 0; y < IMAGE_H; ++y) {
			threads.emplace_back(
				CPU::innerLoop,
				dev,
				IMAGE_W, IMAGE_H,
				filter_size, filter_data_size,
				input_, output_, filter_,
				y);
		}
		for(auto& t : threads) {
			t.join();
		}
	}
}
