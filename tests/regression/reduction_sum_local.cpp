#include "../common.h"

// Parallel reduction sum with local decomposition

// Originally test6

int main() {
  using namespace cl::sycl;

  {
    queue myQueue;

    const auto group_size =
        myQueue.get_device().get_info<info::device::max_work_group_size>();
    const auto size = group_size * 4;

    using type_t = float;

    buffer<type_t> ping(size);
    buffer<type_t> pong(size);

    auto P = &ping;
    auto Q = &pong;

    // Init
    myQueue.submit([&](handler& cgh) {
      auto p = P->get_access<access::mode::write>(cgh);

      cgh.parallel_for<class init>(range<1>(size),
                                   [=](id<1> index) { p[index] = index; });
    });

    size_t local_size = std::min(group_size, size);

    for (size_t N = size; N > 1; N /= local_size) {
      debug() << "Submitting work";
      myQueue.submit([&](handler& cgh) {
        auto input = P->get_access<access::mode::read>(cgh);
        auto output = Q->get_access<access::mode::write>(cgh);

        local_size = std::min(local_size, N);
        auto local =
            accessor<float, 1, access::mode::read_write, access::target::local>(
                local_size, cgh);

        DSELF() << N << local_size;
        cgh.parallel_for<class reduction_sum>(
            nd_range<1>(N / 2, local_size / 2), [=](nd_item<1> index) {
              auto gid = index.get_global(0);
              auto lid = index.get_local(0);
              uint1 N = index.get_global_range().get(0);
              uint1 second = gid + N;

              SYCL_IF(second < 2 * N) {
                local[lid] = input[gid] + input[second];
              }
              SYCL_END;

              index.barrier(access::fence_space::local_space);

              N = min(N, static_cast<uint1>(index.get_local_range().get(0)));

              uint1 stride = N / 2;
              SYCL_WHILE(stride > 0) {
                SYCL_IF(lid < stride) {
                  local[lid] += local[lid + stride];
                }
                SYCL_END;
                index.barrier(access::fence_space::local_space);
                stride /= 2;
              }
              SYCL_END;

              SYCL_IF(lid == 0) {
                output[gid / N] = local[0];
              }
              SYCL_END;
            });
      });

      std::swap(P, Q);
    }

    debug() << "Done, checking results";
    auto p = P->get_access<access::mode::read, access::target::host_buffer>();
    type_t sum =
        (static_cast<type_t>(size) / 2) * static_cast<type_t>(size - 1);
    if (p[0] != sum) {
      debug() << "wrong sum, should be" << sum << "- is" << p[0];
      return 1;
    }
  }

  return 0;
}
