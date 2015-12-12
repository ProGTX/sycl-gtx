#include "../tests.h"

// Random number generation

using cl::sycl::float1;
using cl::sycl::uint1;
using cl::sycl::uint2;

// http://stackoverflow.com/a/16077942
template <class Float, class Uint1, class Uint2>
Float getRandom(Uint2& seed) {
	static const Float invMaxInt = 1.0f / 4294967296.0f;
	Uint1 x = seed.x * 17 + seed.y * 13123;
	seed.x = (x << 13) ^ x;
	seed.y ^= x << 7;
	return (Float)(x * (x * x * 15731 + 74323) + 871483) * invMaxInt;
}

float1 deviceRandom(uint2& seed) {
	using namespace cl::sycl;
	return getRandom<float1, uint1>(seed);
}

float hostRandom(cl_uint2& seed) {
	return getRandom<float, cl_uint>(seed);
}

bool test11() {

	using namespace cl::sycl;
	using namespace std;

	queue myQueue;

	const int size = 4096;
	buffer<float> numbers(size);

	// TODO: Should not be zero
	uint32_t startSeed_x = 24325;
	uint32_t startSeed_y = 32536;

	myQueue.submit([&](handler& cgh) {
		auto n = numbers.get_access<access::mode::discard_write>(cgh);

		cgh.single_task<class generate>([=]() {
			uint2 seed(startSeed_x, startSeed_y);
			SYCL_FOR(int1 i = 0, i < size, ++i) {
				n[i] = deviceRandom(seed);
			}
			SYCL_END
		});
	});

	float eps = 1e-3f;	// Don't need very high accuracy
	auto n = numbers.get_access<access::mode::read, access::target::host_buffer>();
	cl_uint2 seed = { startSeed_x, startSeed_y };

	// TODO: Better automatic testing
	for(auto i = 0; i < size; ++i) {
		auto hostRnd = hostRandom(seed);
		auto deviceRnd = n[i];
		if(deviceRnd < hostRnd - eps || deviceRnd > hostRnd + eps) {
			cout << i << " -> expected " << hostRnd << ", got " << deviceRnd << endl;
			return false;
		}
	}

	return true;
}
