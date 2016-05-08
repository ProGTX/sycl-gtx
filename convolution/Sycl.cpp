#include "Sycl.h"

#include <sycl.hpp>


// TODO: Fix
void Sycl::local(
	int num_invocations, cl_device_id dev,
	int IMAGE_W, int IMAGE_H,
	int filter_size, int filter_data_size,
	const float* input_, float* output_, const float* filter_
) {
	using namespace cl;
	using namespace sycl;

	auto q = queue(device(dev));
	auto data_range = range<1>(IMAGE_W * IMAGE_H);
	auto local_range = range<2>(16, 16);
	int HALF_FILTER_SIZE = filter_size / 2;
	int TWICE_HALF_FILTER_SIZE = HALF_FILTER_SIZE * 2;
	int HALF_FILTER_SIZE_IMAGE_W = HALF_FILTER_SIZE * IMAGE_W;

	int local_mem_size =
		(local_range[0] + TWICE_HALF_FILTER_SIZE) *
		(local_range[1] + TWICE_HALF_FILTER_SIZE);

	auto input_buf = buffer<float4, 1>(
		reinterpret_cast<const sycl::cl_float4*>(input_), data_range);
	auto output_buf = buffer<float4, 1>(
		reinterpret_cast<sycl::cl_float4*>(output_), data_range);
	auto filter_buf = buffer<float4, 1>(
		reinterpret_cast<const sycl::cl_float4*>(filter_),
		range<1>(filter_data_size));

	for(int i = 0; i < num_invocations; ++i) {
		q.submit([&](handler& cgh) {
			auto input = input_buf.get_access<access::mode::read>(cgh);
			auto output = output_buf.get_access<access::mode::discard_write>(cgh);
			auto filter = filter_buf.get_access<access::mode::read>(cgh);
			auto cached = accessor<
				float4, 1, access::mode::read_write, access::target::local>(
					range<1>(local_mem_size));

			cgh.parallel_for<class convolution_local>(
				nd_range<2>(range<2>(IMAGE_W, IMAGE_H), local_range),
				[=](nd_item<2> i)
			{
				auto gid = i.get_global();
				auto lid = i.get_local();
				const int1 rowOffset = gid[1] * IMAGE_W;
				const int1 my = gid[0] + rowOffset;

				const int1 localRowLen =
					TWICE_HALF_FILTER_SIZE + i.get_local_range()[0];
				const int1 localRowOffset = (lid[1] + HALF_FILTER_SIZE) * localRowLen;
				const int1 myLocal = localRowOffset + lid[0] + HALF_FILTER_SIZE;

				// copy my pixel
				cached[myLocal] = input[my];

				SYCL_IF(
					gid[0] < HALF_FILTER_SIZE ||
					gid[0] > IMAGE_W - HALF_FILTER_SIZE - 1 ||
					gid[1] < HALF_FILTER_SIZE ||
					gid[1] > IMAGE_H - HALF_FILTER_SIZE - 1
				) {
					// no computation for me, sync and exit
					i.barrier(access::fence_space::local);
					SYCL_RETURN;
				}
				SYCL_ELSE {
					// copy additional elements
					int1 localColOffset = -1;
					int1 globalColOffset = -1;

					SYCL_IF(lid[0] < HALF_FILTER_SIZE) {
						localColOffset = lid[0];
						globalColOffset = -HALF_FILTER_SIZE;

						cached[localRowOffset + lid[0]] = input[my - HALF_FILTER_SIZE];
					}
					SYCL_ELSE_IF(lid[0] >= i.get_local_range()[0] - HALF_FILTER_SIZE) {
						localColOffset = lid[0] + TWICE_HALF_FILTER_SIZE;
						globalColOffset = HALF_FILTER_SIZE;

						cached[myLocal + HALF_FILTER_SIZE] = input[my + HALF_FILTER_SIZE];
					}
					SYCL_END

					SYCL_IF(lid[1] < HALF_FILTER_SIZE) {
						cached[lid[1] * localRowLen + lid[0] + HALF_FILTER_SIZE] =
							input[my - HALF_FILTER_SIZE_IMAGE_W];
						SYCL_IF(localColOffset > 0) {
							cached[lid[1] * localRowLen + localColOffset] =
								input[my - HALF_FILTER_SIZE_IMAGE_W + globalColOffset];
						}
						SYCL_END
					}
					SYCL_ELSE_IF(lid[1] >= i.get_local_range()[1] - HALF_FILTER_SIZE) {
						int1 offset = (lid[1] + TWICE_HALF_FILTER_SIZE) * localRowLen;
						cached[offset + lid[0] + HALF_FILTER_SIZE] =
							input[my + HALF_FILTER_SIZE_IMAGE_W];
						SYCL_IF(localColOffset > 0) {
							cached[offset + localColOffset] =
								input[my + HALF_FILTER_SIZE_IMAGE_W + globalColOffset];
						}
						SYCL_END
					}
					SYCL_END

					// sync
					i.barrier(access::fence_space::local);

					// perform convolution
					int1 fIndex = 0;
					float4 sum = (float4) 0.0;

					SYCL_FOR(int1 r = -HALF_FILTER_SIZE, r <= HALF_FILTER_SIZE, r++) {
						int1 curRow = r * localRowLen;
						SYCL_FOR(int1 c = -HALF_FILTER_SIZE, c <= HALF_FILTER_SIZE, c++) {
							sum += cached[myLocal + curRow + c] * filter[fIndex];
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

	for(int i = 0; i < num_invocations; ++i) {
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
