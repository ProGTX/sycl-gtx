#include "../tests.h"

// Test vectors in kernel

#include <sstream>

template <class T>
std::string to_string(const T& t) {
	std::stringstream s;
	s << '(' << t.x() << ',' << t.y() << ',' << t.z() << ')';
	return s.str();
}

bool test10() {

	using namespace cl::sycl;
	using namespace std;

	cpu_selector gpu;
	queue myQueue(gpu);

	const int size = 10;
	cl::sycl::cl_float3 testVector;
	testVector.x() = 1;
	testVector.x() = 2;
	testVector.x() = 3;
	buffer<float3> vectors(size);

	myQueue.submit([&](handler& cgh) {
		auto v = vectors.get_access<access::mode::discard_write>(cgh);
			
		cgh.parallel_for<class addition>(range<1>(size), [=](id<> i) {
			v[i] = float3(testVector.x(), testVector.y(), testVector.z());
		});
	});

	auto v = vectors.get_access<access::mode::read, access::target::host_buffer>();

	for(auto i = 0; i < size; ++i) {
		auto vi = v[i];
		if(vi.x() != 1 || vi.y() != 2 || vi.z() != 3) {
			cout << i << " -> expected " << to_string(testVector) << ", got " << to_string(vi) << endl;
			return false;
		}
	}

	return true;
}
