#include "tests.h"

// Parallel reduction sum

bool test4() {
	static const size_t N = 1024;

	using namespace cl::sycl;

	{
		queue myQueue;

		buffer<int> V(N);

		command_group(myQueue, [&]() {
			auto v = V.get_access<access::read_write>();

			// Init
			parallel_for<>(range<1>(N), [=](id<1> index) {
				v[index] = index;
			});

			// Calculate reduction sum
			for(size_t offset = 1; offset < N; offset *= 2) {
				parallel_for<>(range<1>(N), [=](id<1> index) {
					if(index % (2 * offset) == 0 && index + offset < N) {
						v[index] += v[index + offset];
					}
				});
			}
		});

		auto v = V.get_access<access::read, access::host_buffer>();

		int sum = 0;
		for(size_t i = 0; i < N; ++i) {
			sum += i;

			auto vi = v[i];
			if(vi != sum) {
				debug() << i << "expected" << sum << "actual" << vi;
				return false;
			}
		}
	}

	return true;
}
