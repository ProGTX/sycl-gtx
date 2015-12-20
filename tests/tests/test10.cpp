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
	testVector.y() = 2;
	testVector.z() = 3;
	buffer<float3> vectors(size);

	myQueue.submit([&](handler& cgh) {
		auto v = vectors.get_access<access::mode::discard_write>(cgh);
			
		cgh.parallel_for<class addition>(range<1>(size), [=](id<> i) {
			v[i] = float3(testVector.x(), testVector.y(), testVector.z());
		});
	});

	auto v = vectors.get_access<access::mode::read, access::target::host_buffer>();

	auto floatEqual = [](float& first, float& second) {
		static const double eps = 1e5f;
		return first > second - eps && first < second + eps;
	};

	for(auto i = 0; i < size; ++i) {
		auto vi = v[i];
		if(	!floatEqual(vi.x(), testVector.x()) ||
			!floatEqual(vi.y(), testVector.y()) ||
			!floatEqual(vi.z(), testVector.z())
		) {
			cout << i << " -> expected " << to_string(testVector) << ", got " << to_string(vi) << endl;
			return false;
		}
	}

	return true;
}
