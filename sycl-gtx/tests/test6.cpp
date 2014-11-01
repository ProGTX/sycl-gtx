#include "tests.h"

bool test6() {
	using namespace cl::sycl;

	cl::sycl::queue q;

	auto a = cl::sycl::exception(&q, 0);
	auto q2 = a.get_queue();
	auto img = a.get_image();

	debug("&q\t") << &q;
	debug("q2\t") << q2;
	debug("img\t") << img;

	int bb[5];
	cl::sycl::buffer<int, 3> b(bb, 4);
	auto a2 = cl::sycl::exception(&b, 0);

	return true;
}
