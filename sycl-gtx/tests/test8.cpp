#include "tests.h"

// 2.6 Anatomy of a SYCL application
// (slightly revised from revision 2014-09-16 of the specification)

bool test8() {
	using namespace cl::sycl;

	int data[1024]; // initialize data to be worked on

	// By including all the SYCL work in a {} block, we ensure
	// all SYCL tasks must complete before exiting the block
	{
		// create a queue to enqueue work to
		queue myQueue;

		// wrap our data variable in a buffer
		buffer<int, 1> resultBuf(data, 1024);

		// create a command_group to issue commands to the queue
		command_group(myQueue, [&]() {
			// request access to the buffer
			auto writeResult = resultBuf.get_access<access::write>();

			// enqueue a prallel_for task
			parallel_for<>(range<1>(1024), [=](id<1> idx) {
				writeResult[idx] = idx;
			}); // end of the kernel function
		}); // end of our commands for this queue

	} // end of scope, so we wait for the queued work to complete

	// print result
	for(int i = 0; i < 1024; i++) {
		printf("data[%d] = %d\n", i, data[i]);
	}

	return true;
}
