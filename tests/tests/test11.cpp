#include "../tests.h"

// Random number generation

using cl::sycl::float1;
using cl::sycl::uint1;
using cl::sycl::uint2;

// http://stackoverflow.com/a/16077942
float1 getRandom(uint2& seed) {
	static const float1 invMaxInt = 1.0f / 4294967296.0f;
	uint1 x = seed.x * 17 + seed.y * 13123;
	seed.x = (x << 13) ^ x;
	seed.y ^= x << 7;
	return (float1)(x * (x * x * 15731 + 74323) + 871483) * invMaxInt;
}

bool test11() {

	using namespace cl::sycl;
	using namespace std;

	queue myQueue;

	const int size = 4096;
	buffer<float> numbers(size);

	myQueue.submit([&](handler& cgh) {
		auto n = numbers.get_access<access::discard_write>(cgh);

		cgh.single_task<class generate>([=]() {
			uint2 seed;
			SYCL_FOR(int1 i = 0, i < size, ++i)
			SYCL_BEGIN {
				n[i] = getRandom(seed);
			}
			SYCL_END
		});
	});

	auto n = numbers.get_access<access::read, access::host_buffer>();

	// TODO: How to automatically check correctness?
	for(auto i = 0; i < size; ++i) {
		cout << n[i] << ", ";
	}
	cout << endl;

	return true;
}
