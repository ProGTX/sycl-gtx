#include "../tests.h"

// Parallel reduction sum

bool test4() {
	static const size_t N = 1024;

	using namespace cl::sycl;

	{
		queue myQueue;

		buffer<int> V(N);

		myQueue.submit([&](handler& cgh) {
			auto v = V.get_access<access::mode::read_write>(cgh);

			// Init
			cgh.parallel_for<class init>(range<1>(N), [=](id<1> index) {
				v[index] = index;
			});

			// Calculate reduction sum
			for(size_t stride = 1; stride < N; stride *= 2) {
				cgh.parallel_for<class reduction_sum>(range<1>(N / 2 / stride), [=](id<1> index) {
					auto i = 2 * stride * index;
					v[i] += v[i + stride];
				});
			}
		});

		auto v = V.get_access<access::mode::read, access::target::host_buffer>();
		int sum = ((uint64_t)N * (uint64_t)(N - 1)) / 2;

		if(v[0] != sum) {
			debug() << "wrong sum, should be" << sum << "- is" << v[0];
				return false;
			}
		}

	return true;
}
