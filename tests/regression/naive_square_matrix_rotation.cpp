#include "../common.h"

// Naive square matrix rotation

// Originally test3

using namespace cl::sycl;

// Size of the square matrices
#ifndef NDEBUG
const size_t N = 128;
#else
const size_t N = 1024;
#endif

int main() {
  {
    queue myQueue;

    buffer<float, 2> A(range<2>(N, N));
    buffer<float, 2> B(range<2>(N, N));

    debug() << "Initializing buffer A";
    {
      auto ah =
          A.get_access<access::mode::read_write, access::target::host_buffer>();
      for (int i = 0; i < N; ++i) {
        for (int j = 0; j < N; ++j) {
          ah[i][j] = static_cast<float>(i + j * N);
        }
      }
    }

    // Rotate A and store result to B
    debug() << "Submitting work";
    myQueue.submit([&](handler& cgh) {
      auto a = A.get_access<access::mode::read>(cgh);
      auto b = B.get_access<access::mode::write>(cgh);

      cgh.parallel_for<class rotation>(
          range<2>(N, N), [=](id<2> i) { b[N - i[1] - 1][i[0]] = a[i]; });
    });

    debug() << "Done, checking results";
    auto ah =
        A.get_access<access::mode::read_write, access::target::host_buffer>();
    auto bh = B.get_access<access::mode::read, access::target::host_buffer>();
    for (int i = 0; i < N; ++i) {
      for (int j = 0; j < N; ++j) {
        auto expected = ah[i][j];
        auto actual = bh[N - j - 1][i];
        if (actual != expected) {
          debug() << i << j << "expected" << expected << "actual" << actual;
          return 1;
        }
      }
    }
  }

  return 0;
}
