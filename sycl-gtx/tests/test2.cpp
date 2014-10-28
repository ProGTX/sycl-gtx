#include "tests.h"

// 2.5.6 Anatomy of a SYCL application

bool test2() {
	using namespace cl::sycl;

	int result; // this is where we will write our result

	{	// by sticking all the SYCL work in a {} block, we ensure
		// all SYCL tasks must complete before exiting the block

		// create a queue to work on
		queue myQueue;

		// wrap our result variable in a buffer
		buffer<int> resultBuf(&result, 1);

		// create some commands for our queue
		command_group(myQueue, [&]() {
			// request access to our buffer
			auto writeResult = resultBuf.get_access<access::write>();

			// enqueue a single, simple task
			single_task(kernel_lambda<class simple_test>([=]() {
				writeResult[0] = 1234;
			}))
		}); // end of our commands for this queue

	} // end scope, so we wait for the queue to complete

	printf("Result = %d\n", result);

	return true;
}
