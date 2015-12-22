#include "../tests.h"

// Check access into SYCL CL types


bool test12() {
	using namespace std;
	using namespace cl::sycl;

	auto size = 1;
	auto spheres = buffer<int8>(range<1>(size));

	::cl_int8 testVector = {
		0, 1, 2, 3, 4, 5, 6, 7
	};

	{
		auto s = spheres.get_access<access::mode::discard_write, access::target::host_buffer>();

		// See SphereSycl
		for(int i = 0; i < size; ++i) {
			auto& si = s[i];

			si.lo().x() = testVector.s0 * (i + 1);
			si.lo().y() = testVector.s1 * (i + 1);
			si.lo().z() = testVector.s2 * (i + 1);
			si.lo().w() = testVector.s3 * (i + 1);

			si.hi().x() = testVector.s4 * (i + 1);
			si.hi().y() = testVector.s5 * (i + 1);
			si.hi().z() = testVector.s6 * (i + 1);
			si.hi().w() = testVector.s7 * (i + 1);
		}
	}

	{
		auto s = spheres.get_access<access::mode::read, access::target::host_buffer>();

		auto compare = [](::cl_int expected, ::cl_int actual) {
#ifdef _DEBUG
			cout << expected << ", " << actual << endl;
#endif
			if(actual != expected) {
				cout << "Wrong output, expected " << expected << ", got " << actual << endl;
				return false;
			}
			return true;
		};

		// See SphereSycl
		for(int i = 0; i < size; ++i) {
			auto& si = s[i];

			if(!compare(testVector.s0, si.lo().x())) {
				return false;
			}
			if(!compare(testVector.s1, si.lo().y())) {
				return false;
			}
			if(!compare(testVector.s2, si.lo().z())) {
				return false;
			}
			if(!compare(testVector.s3, si.lo().w())) {
				return false;
			}

			if(!compare(testVector.s4, si.hi().x())) {
				return false;
			}
			if(!compare(testVector.s5, si.hi().y())) {
				return false;
			}
			if(!compare(testVector.s6, si.hi().z())) {
				return false;
			}
			if(!compare(testVector.s7, si.hi().w())) {
				return false;
			}
		}
	}

	return true;
}
