#include "tests.h"

#include <sstream>

template <class T>
std::string to_string(const T& t) {
	std::stringstream s;
	s << '(' << t.x << ',' << t.y << ',' << t.z << ')';
	return s.str();
}

bool test10() {

	using namespace cl::sycl;
	using namespace std;

	queue myQueue;
	const int size = 10;
	cl_double3 testVector = { 1, 2, 3 };
	buffer<double3> vectors(size);

	myQueue.submit([&](handler& cgh) {
		auto v = vectors.get_access<access::discard_write>(cgh);
			
		cgh.parallel_for<class addition>(range<1>(size), [=](id<> i) {
			v[i] = double3(testVector.x, testVector.y, testVector.z);
		});
	});

	auto v = vectors.get_access<access::read, access::host_buffer>();

	for(auto i = 0; i < size; ++i) {
		auto vi = v[i];
		if(vi.x != 1 || vi.y != 2 || vi.z != 3) {
			cout << i << " -> expected " << to_string(testVector) << ", got " << to_string(vi) << endl;
			return false;
		}
	}

	return true;
}
