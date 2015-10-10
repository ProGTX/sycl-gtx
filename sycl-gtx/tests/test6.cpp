#include "tests.h"

// Parallel reduction sum with local decomposition

bool test6() {
	using namespace cl::sycl;

	{
		queue myQueue;

		const auto group_size = myQueue.get_device().get_info<info::device::max_work_group_size>();
		const auto size = group_size * 2;	// TODO: Bigger data

		using type_t = float;

		buffer<type_t> ping(size);
		buffer<type_t> pong(size);

		auto P = &ping;
		auto Q = &pong;

		// Init
		myQueue.submit([&](handler& cgh) {
			auto p = P->get_access<access::write>(cgh);

			cgh.parallel_for<>(range<1>(size), [=](id<1> index) {
				p[index] = index;
			});
		});

		for(unsigned int N = size / 2; N > 0; N /= 2 * group_size) {
			debug() << "Submitting work";
			myQueue.submit([&](handler& cgh) {
				auto input = P->get_access<access::read>(cgh);
				auto output = Q->get_access<access::write>(cgh);
				auto local = accessor<float, 1, access::read_write, access::local>(group_size);

				cgh.parallel_for<>(nd_range<1>(N, group_size), [=](nd_item<1> index) {
					auto gid = index.get_global(0);
					auto lid = index.get_local(0);
					uint1 N = index.get_global_range().get(0);
					uint1 second = gid + N;

					SYCL_IF(second < 2 * N)
					SYCL_THEN({
						local[lid] = input[gid] + input[second];
					})

					index.barrier(access::fence_space::local);

					N = min(N, (uint1)index.get_local_range().get(0));

					uint1 stride = N / 2;
					SYCL_WHILE(stride > 0)
					SYCL_BLOCK({
						SYCL_IF(lid < stride)
						SYCL_THEN({
							local[lid] += local[lid + stride];
						})
						index.barrier(access::fence_space::local);
						stride /= 2;
					})

					SYCL_IF(lid == 0)
					SYCL_THEN({
						output[gid / N] = local[0];
					})
				});
			});

			std::swap(P, Q);
		}

		debug() << "Done, checking results";
		auto p = P->get_access<access::read, access::host_buffer>();
		type_t sum = ((type_t)size / 2) * (type_t)(size - 1);
		if(p[0] != sum) {
			debug() << "wrong sum, should be" << sum << "- is" << p[0];
			return false;
		}
	}

	return true;
}
