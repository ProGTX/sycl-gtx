#include "../common.h"

// 2.12 Example SYCL application

// Originally test7

using namespace cl::sycl;

// Size of the matrices
// Note: Checking results at end can be very slow
// - this size is still manageable if optimizations are on
#ifndef NDEBUG
const size_t N = 200;
const size_t M = 100;
#else
const size_t N = 2000;
const size_t M = 1000;
#endif

int main() {
  {  // By including all the SYCL work in a {} block,
    // we ensure all SYCL tasks must complete before exiting the block

    // Create a queue to work on
    queue myQueue;

    // Create some 2D buffers of float for our matrices
    buffer<float, 2> a(range<2>(N, M));
    buffer<float, 2> b(range<2>(N, M));
    buffer<float, 2> c(range<2>(N, M));

    // Launch a first asynchronous kernel to initialize a
    myQueue.submit([&](handler& cgh) {
      // The kernel writes a, so get a write accessor on it
      auto A = a.get_access<access::mode::write>(cgh);

      // Enqueue a parallel kernel iterating on a N*M 2D iteration space
      cgh.parallel_for<class init_a>(range<2>(N, M), [=](id<2> index) {
        A[index] = index[0] * 2 + index[1];
      });
    });

    // Launch an asynchronous kernel to initialize b
    myQueue.submit([&](handler& cgh) {
      // The kernel write b, so get a write accessor on it
      auto B = b.get_access<access::mode::write>(cgh);
      // From the access pattern above,
      // the SYCL runtime detect this command_group is independent from the
      // first one
      // and can be scheduled independently

      // Enqueue a parallel kernel iterating on a N*M 2D iteration space
      cgh.parallel_for<class init_b>(range<2>(N, M), [=](id<2> index) {
        B[index] = index[0] * 2014 + index[1] * 42;
      });
    });

    // Launch an asynchronous kernel to compute matrix addition c = a + b
    myQueue.submit([&](handler& cgh) {
      // In the kernel a and b are read, but c is written
      auto A = a.get_access<access::mode::read>(cgh);
      auto B = b.get_access<access::mode::read>(cgh);
      auto C = c.get_access<access::mode::write>(cgh);
      // From these accessors, the SYCL runtime will ensure that when
      // this kernel is run, the kernels computing a and b completed

      // Enqueue a parallel kernel iterating on a N*M 2D iteration space
      cgh.parallel_for<class matrix_add>(
          range<2>(N, M), [=](id<2> index) { C[index] = A[index] + B[index]; });
    });

    debug() << "Done, checking results";
    // Ask for access to read c from the host-side.
    // The SYCL runtime ensures that c is ready when the accessor is returned
    auto C = c.get_access<access::mode::read, access::target::host_buffer>();
    for (int i = 0; i < N; ++i) {
      for (int j = 0; j < M; ++j) {
        // Compare the result to the analytic value
        auto expected = i * (2 + 2014) + j * (1 + 42);
        if (C[i][j] != expected) {
          debug() << i << j << "expected" << expected << "actual" << C[i][j];
          return 1;
        }
      }
    }

  }  // End scope of myQueue,
  // which waits for any remaining operations on the queue to complete

  debug() << "Good computation!";
  return 0;
}
