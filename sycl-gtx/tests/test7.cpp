#include "tests.h"

// 2.12 Example SYCL application

using namespace cl::sycl;

// Size of the matrices
const size_t N = 2000;
const size_t M = 3000;

bool test7() {
	{ // By including all the SYCL work in a {} block, we ensure
		// all SYCL tasks must complete before exiting the block

		// Create a queue to work on
		queue myQueue;

		// Create some 2D buffers of float for our matrices
		buffer<float, 2> a({ N, M });
		buffer<float, 2> b({ N, M });
		buffer<float, 2> c({ N, M });

		// Launch a first asynchronous kernel to initialize a
		command_group(myQueue, [&]() {
			// The kernel write a, so get a write accessor on it
			auto A = a.get_access<access::write>();

			// Enqueue a parallel kernel iterating on a N*M 2D iteration space
			parallel_for(N*M,//<class init_a>(range<2>(N, M),
				[=](id<2> index) {
				//A[index] = index[0] * 2 + index[1];
			});
		});

		// Launch an asynchronous kernel to initialize b
		command_group(myQueue, [&]() {
			// The kernel write b, so get a write accessor on it
			auto B = b.get_access<access::write>();
			/* From the access pattern above, the SYCL runtime detect this
			command_group is independant from the first one and can be
			scheduled independently */

			// Enqueue a parallel kernel iterating on a N*M 2D iteration space
			parallel_for(N*M,//<class init_b>(range<2>(N, M),
				[=](id<2> index) {
				//B[index] = index[0] * 2014 + index[1] * 42;
			});
		});

		// Launch an asynchronous kernel to compute matrix addition c = a + b
		command_group(myQueue, [&]() {
			// In the kernel a and b are read, but c is written
			auto A = a.get_access<access::read>();
			auto B = b.get_access<access::read>();
			auto C = c.get_access<access::write>();
			// From these accessors, the SYCL runtime will ensure that when
			// this kernel is run, the kernels computing a and b completed

			// Enqueue a parallel kernel iterating on a N*M 2D iteration space
			parallel_for(N*M,//<class matrix_add>(range<2>(N, M),
				[=](id<2> index) {

				//C[index] = A[index] + B[index];
			});
		});

		/* Ask an access to read c from the host-side. The SYCL runtime
		ensures that c is ready when the accessor is returned */
		auto C = c.get_access<access::read, access::host_buffer>();
		std::cout << std::endl << "Result:" << std::endl;
		for(size_t i = 0; i < N; i++) {
			for(size_t j = 0; j < M; j++) {
				// Compare the result to the analytic value
				/*
				if(C[i][j] != i*(2 + 2014) + j*(1 + 42)) {
				std::cout << "Wrong value " << C[i][j] << " on element "
				<< i << " " << j << std::endl;
				exit(-1);
				}
				*/
			}
		}

	} /* End scope of myQueue, this wait for any remaining operations on the
	  queue to complete */

	std::cout << "Good computation!" << std::endl;
	return true;
}
