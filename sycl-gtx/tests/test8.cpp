#include "tests.h"

// 2.6 Anatomy of a SYCL application
// (slightly revised from revision 2014-09-16 of the specification)

bool test8() {
	using namespace cl::sycl;

	static const int size = 1024;
	int data[size];		// Initialize data to be worked on

	// Expected results
	int expected[size];
	for(int i = 0; i < size; ++i) {
		expected[i] = i;
	}

	// By including all the SYCL work in a {} block, we ensure
	// all SYCL tasks must complete before exiting the block
	{
		// create a queue to enqueue work to
		queue myQueue;

		// wrap our data variable in a buffer
		buffer<int, 1> resultBuf(data, size);

		// create a command_group to issue commands to the queue
		myQueue.submit([&]() {
			// request access to the buffer
			auto writeResult = resultBuf.get_access<access::write>();

			// enqueue a prallel_for task
			parallel_for</*class simple_test*/>(range<1>(size), [=](id<1> idx) {
				writeResult[idx] = idx;
			}); // end of the kernel function
		}); // end of our commands for this queue

	} // end of scope, so we wait for the queued work to complete

	bool success = true;

	// Print result
	for(int i = 0; i < size; ++i) {
		if(data[i] != expected[i]) {
			debug() << i << ",\texpected" << expected[i] << ",\tactual" << data[i];
			success = false;
		}
	}

	return success;
}
