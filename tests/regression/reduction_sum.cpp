#include "../common.h"

// Parallel reduction sum

// Originally test4

int main() {
  static const size_t N = 1024;

  using namespace cl::sycl;

  {
    queue myQueue;

    buffer<int> V(N);
    buffer<int> stride_tmp(1);
    {
      auto s = stride_tmp.get_access<access::mode::discard_write,
                                     access::target::host_buffer>();
      s[0] = 1;
    }

    myQueue.submit([&](handler& cgh) {
      auto v = V.get_access<access::mode::read_write>(cgh);

      // Init
      cgh.parallel_for<class init>(range<1>(N),
                                   [=](id<1> index) { v[index] = index; });

      auto s = stride_tmp.get_access<access::mode::read_write>(cgh);

      // Calculate reduction sum
      for (size_t stride = 1; stride < N; stride *= 2) {
        cgh.parallel_for<class reduction_sum>(range<1>(N / 2 / stride),
                                              [=](id<1> index) {
                                                auto i = 2 * s[0] * index;
                                                v[i] += v[i + s[0]];
                                                s[0] *= 2;
                                              });
      }
    });

    auto v = V.get_access<access::mode::read, access::target::host_buffer>();
    int sum = (static_cast<uint64_t>(N) * (uint64_t)(N - 1)) / 2;  // NOLINT

    if (v[0] != sum) {
      debug() << "wrong sum, should be" << sum << "- is" << v[0];
      return 1;
    }
  }

  return 0;
}
