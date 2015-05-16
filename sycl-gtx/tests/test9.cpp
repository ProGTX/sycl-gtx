#include "tests.h"

// Work efficient prefix sum

bool test9() {
	using namespace cl::sycl;

	{
		queue myQueue;

		const auto group_size = myQueue.get_device().get_info<CL_DEVICE_MAX_WORK_GROUP_SIZE>();
		const auto size = group_size * 16;

		buffer<float> ping(size);
		buffer<float> pong(size);

		auto P = &ping;
		auto Q = &pong;

		// Init
		command_group(myQueue, [&]() {
			auto p = P->get_access<access::write>();

			parallel_for<>(range<1>(size), [=](id<1> index) {
				p[index] = index;
			});
		});

		// TODO: Computation

		auto p = P->get_access<access::read, access::host_buffer>();

		float sum = 0;
		for(size_t i = 0; i < size; ++i) {
			sum += (float)i;
			if(p[i] != sum) {
				debug() << "wrong sum, should be" << sum << "- is" << p[i];
				return false;
			}
		}
	}

	return true;
}
