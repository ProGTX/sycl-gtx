#include "tests.h"

// Naive square matrix rotation

using namespace cl::sycl;

// Size of the square matrices
const size_t N = 2000;

bool test3() {
	{
		queue myQueue;

		buffer<float, 2> A({ N, N });
		buffer<float, 2> B({ N, N });

		// Init A
		auto ah = A.get_access<access::read_write, access::host_buffer>();
		for(int i = 0; i < N; ++i) {
			for(int j = 0; j < N; ++j) {
				ah[i][j] = i + j * N;
			}
		}

		// Rotate A and store result to B
		command_group(myQueue, [&]() {
			auto a = A.get_access<access::read>();
			auto b = B.get_access<access::write>();

			parallel_for<>(range<2>(N, N), [=](id<2> i) {
				b[N - i[1] - 1][i[0]] = a[i];
			});
		});

		// Check result
		auto bh = B.get_access<access::read, access::host_buffer>();
		for(int i = 0; i < N; ++i) {
			for(int j = 0; j < N; ++j) {
				auto expected = ah[N - j - 1][i];
				auto actual = bh[i][j];
				if(actual != expected) {
					debug() << i << j << "expected" << expected << "actual" << actual;
					return false;
				}
			}
		}
	}

	return true;
}
