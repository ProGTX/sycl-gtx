#include "Sycl.h"

#include <sycl.hpp>


void Sycl::global(
	int num_invocations, cl_device_id dev,
	int IMAGE_W, int IMAGE_H,
	int filter_size, int filter_data_size,
	const float* input_, float* output_, const float* filter_
) {
	using namespace cl;
	using namespace sycl;

	auto q = queue(device(dev));
	auto global_range = range<2>(IMAGE_W, IMAGE_H);
	auto data_range = range<1>(IMAGE_W * IMAGE_H);
	int HALF_FILTER_SIZE = filter_size / 2;

	auto input_buf = buffer<float4, 1>(
		reinterpret_cast<const sycl::cl_float4*>(input_), data_range);
	auto output_buf = buffer<float4, 1>(
		reinterpret_cast<sycl::cl_float4*>(output_), data_range);
	auto filter_buf = buffer<float4, 1>(
		reinterpret_cast<const sycl::cl_float4*>(filter_),
		range<1>(filter_data_size));

	for (int i = 0; i < num_invocations; ++i) {
		q.submit([&](handler& cgh) {
			auto input = input_buf.get_access<access::mode::read>(cgh);
			auto output = output_buf.get_access<access::mode::discard_write>(cgh);
			auto filter = filter_buf.get_access<access::mode::read>(cgh);

			cgh.parallel_for<class convolution_global>(global_range, [=](id<2> i) {
				int1 rowOffset = i[1] * IMAGE_W;
				int1 my = i[0] + rowOffset;

				SYCL_IF(
					i[0] < HALF_FILTER_SIZE ||
					i[0] > IMAGE_W - HALF_FILTER_SIZE - 1 ||
					i[1] < HALF_FILTER_SIZE ||
					i[1] > IMAGE_H - HALF_FILTER_SIZE - 1
				) {
					SYCL_RETURN;
				}
				SYCL_ELSE {
					// perform convolution
					int1 fIndex = 0;
					float4 sum = (float4) 0.0;

					SYCL_FOR(int1 r = -HALF_FILTER_SIZE, r <= HALF_FILTER_SIZE, r++) {
						int1 curRow = my + r * IMAGE_W;
						SYCL_FOR(int1 c = -HALF_FILTER_SIZE, c <= HALF_FILTER_SIZE, c++) {
							sum += input[curRow + c] * filter[fIndex];
							fIndex += 1;
						}
						SYCL_END
					}
					SYCL_END
					output[my] = sum;
				}
				SYCL_END
			});
		});
		q.wait();
	}
}
