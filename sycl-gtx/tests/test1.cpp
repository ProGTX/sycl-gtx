#include "tests.h"

#include <vector>

// Example from http://www.codeplay.com/portal/sycl-tutorial-1-the-vector-addition
// (slightly modified)

#define TOL (0.001)   // tolerance used in floating point comparisons
#define LENGTH (1024) // Length of vectors a, b and c

bool test1() {
	using namespace cl::sycl;

	std::vector<int> h_a(LENGTH);             // a vector
	std::vector<int> h_b(LENGTH);             // b vector
	std::vector<int> h_c(LENGTH);             // c vector
	std::vector<int> h_r(LENGTH, 0xdeadbeef); // d vector (result)
	// Fill vectors a and b with random float values
	int count = LENGTH;
	for(int i = 0; i < count; i++) {
		h_a[i] = (int) (rand() / (float)RAND_MAX);
		h_b[i] = (int)(rand() / (float)RAND_MAX);
		h_c[i] = (int)(rand() / (float)RAND_MAX);
	}
	{
		// Device buffers
		buffer<int> d_a(h_a);
		buffer<int> d_b(h_b);
		buffer<int> d_c(h_c);
		buffer<int> d_r(h_d);
		queue myQueue;
		command_group(myQueue, [&]() {
			// Data accessors
			auto a = d_a.get_access<access::read>();
			auto b = d_b.get_access<access::read>();
			auto c = d_c.get_access<access::read>();
			auto r = d_r.get_access<access::write>();
			// Kernel
			parallel_for(count, kernel_functor([=](id<> item) {
				int i = item.get_global(0);
				r[i] = a[i] + b[i] + c[i];
			}));
		});
	}
	// Test the results
	int correct = 0;
	float tmp;
	for(int i = 0; i < count; i++) {
		tmp = h_a[i] + h_b[i] + h_c[i]; // assign element i of a+b+c to tmp
		tmp -= h_r[i]; // compute deviation of expected and output result
		if(tmp * tmp < TOL * TOL) // correct if square deviation is less than
			// tolerance squared
		{
			correct++;
		}
		else {
			printf(" tmp %f h_a %f h_b %f h_c %f h_r %f \n", tmp, h_a[i], h_b[i],
				h_c[i], h_r[i]);
		}
	}
	// summarize results
	printf("R = A+B+C: %d out of %d results were correct.\n", correct, count);

	return (correct == count);
}

#undef TOL
#undef LENGTH
