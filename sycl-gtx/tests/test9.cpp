#include "tests.h"

// Work efficient prefix sum

bool test9() {
	using namespace cl::sycl;

	{
		queue myQueue;

		const auto group_size = myQueue.get_device().get_info<CL_DEVICE_MAX_WORK_GROUP_SIZE>();
		const auto size = group_size * 2;

		buffer<float> data(size);

		// Init
		command_group(myQueue, [&]() {
			auto d = data.get_access<access::write>();

			parallel_for<>(range<1>(size), [=](id<1> index) {
				d[index] = index;
			});
		});

		command_group(myQueue, [&]() {
			auto input = data.get_access<access::read_write>();
			auto localBlock = accessor<float, 1, access::read_write, access::local>(group_size);

			// TODO: Extend for large arrays
			parallel_for<>(nd_range<1>(size / 2, group_size), [=](nd_item<1> index) {
				uint1 GID = 2 * index.get_global_id(0);
				uint1 LID = 2 * index.get_local_id(0);

				localBlock[LID]		= input[GID];
				localBlock[LID + 1]	= input[GID + 1];

				index.barrier(access::fence_space::local);

				uint1 N = 2 * index.get_local_size(0);
				uint1 first;
				uint1 second;

				LID /= 2;

				// Up-sweep
				uint1 offset = 1;
				SYCL_WHILE(offset < N)
				SYCL_BEGIN {
					SYCL_IF(LID % offset == 0)
					SYCL_BEGIN {
						first = 2 * LID + offset - 1;
						second = first + offset;
						localBlock[second] = localBlock[first] + localBlock[second];
					}
					SYCL_END

					index.barrier(access::fence_space::local);
					offset *= 2;
				}
				SYCL_END

				SYCL_IF(LID == 0) 
				SYCL_THEN({
					localBlock[N - 1] = 0;
				})
				index.barrier(access::fence_space::local);

				uint1 tmp;

				// Down-sweep
				offset = N;
				SYCL_WHILE(offset > 0)
				SYCL_BEGIN {
					SYCL_IF(LID % offset == 0)
					SYCL_BEGIN {
						first = 2 * LID + offset - 1;
						second = first + offset;
						tmp = localBlock[second];
						localBlock[second] = localBlock[first] + tmp;
						localBlock[first] = tmp;
					}
					SYCL_END

					index.barrier(access::fence_space::local);
					offset /= 2;
				}
				SYCL_END

				LID *= 2;
				input[GID]		+= localBlock[LID];
				input[GID + 1]	+= localBlock[LID + 1];
			});
		});

		auto d = data.get_access<access::read, access::host_buffer>();

		float sum = 0;
		for(size_t i = 0; i < size; ++i) {
			sum += (float)i;
			if(d[i] != sum) {
				debug() << "wrong sum, should be" << sum << "- is" << d[i];
				return false;
			}
		}
	}

	return true;
}
