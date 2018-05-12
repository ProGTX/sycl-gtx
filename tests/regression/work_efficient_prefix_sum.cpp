#include "../common.h"

// Work efficient prefix sum

// Originally test9

template <typename F, typename std::enable_if<
                          std::is_floating_point<F>::value>::type* = nullptr>
bool check_sum(cl::sycl::buffer<F, 1>& data) {
  using namespace cl::sycl;

  auto d = data.template get_access<access::mode::read,
                                    access::target::host_buffer>();
  F sum = 0;
  for (size_t i = 0; i < data.get_count(); ++i) {
    sum += static_cast<F>(1);
    auto diff = std::abs(sum - d[i]);
    if (diff > 0.01) {
      debug() << "wrong sum, should be" << sum << "- is" << d[i];
      return false;
    }
  }

  return true;
}

template <typename I,
          typename std::enable_if<std::is_integral<I>::value>::type* = nullptr>
bool check_sum(cl::sycl::buffer<I, 1>& data) {
  using namespace cl::sycl;

  auto d = data.template get_access<access::mode::read,
                                    access::target::host_buffer>();
  I sum = 0;
  for (size_t i = 0; i < data.get_count(); ++i) {
    sum += static_cast<I>(1);
    if (d[i] != sum) {
      debug() << "wrong sum, should be" << sum << "- is" << d[i];
      return false;
    }
  }

  return true;
}

template <typename T>
struct prefix_sum_kernel {
 protected:
  using mode = cl::sycl::access::mode;
  using target = cl::sycl::access::target;
  using buffer = cl::sycl::buffer<T>;

  // Constants
  size_t global_size;

  // Global memory
  cl::sycl::accessor<T> input;
  cl::sycl::accessor<T, 1, mode::write> higher_level;

  // Local memory
  cl::sycl::accessor<T, 1, mode::read_write, target::local> localBlock;

 public:
  prefix_sum_kernel(cl::sycl::handler& cgh, buffer& data, buffer& higher_level,
                    size_t global_size, size_t local_size)
      : input(data.template get_access<mode::read_write>(cgh)),
        higher_level(higher_level.template get_access<mode::write>(cgh)),
        localBlock(local_size, cgh),
        global_size(global_size) {}

  void operator()(cl::sycl::nd_item<1> index) {
    using namespace cl::sycl;

    uint1 GID = 2 * index.get_global(0);
    uint1 LID = 2 * index.get_local(0);

    SYCL_IF(GID < global_size) {
      localBlock[LID] = input[GID];
      localBlock[LID + 1] = input[GID + 1];
    }
    SYCL_END;

    index.barrier(access::fence_space::local_space);

    uint1 local_size = 2 * index.get_local_range()[0];
    uint1 first;
    uint1 second;

    LID /= 2;

    // Up-sweep
    uint1 offset = 1;
    SYCL_WHILE(offset < local_size) {
      SYCL_IF(LID % offset == 0) {
        first = 2 * LID + offset - 1;
        second = first + offset;
        localBlock[second] = localBlock[first] + localBlock[second];
      }
      SYCL_END;

      index.barrier(access::fence_space::local_space);
      offset *= 2;
    }
    SYCL_END;

    SYCL_IF(LID == 0) {
      localBlock[local_size - 1] = 0;
    }
    SYCL_END;
    index.barrier(access::fence_space::local_space);

    vec<T, 1> tmp;

    // Down-sweep
    offset = local_size;
    SYCL_WHILE(offset > 0) {
      SYCL_IF(LID % offset == 0) {
        first = 2 * LID + offset - 1;
        second = first + offset;
        tmp = localBlock[second];
        localBlock[second] = localBlock[first] + tmp;
        localBlock[first] = tmp;
      }
      SYCL_END;

      index.barrier(access::fence_space::local_space);
      offset /= 2;
    }
    SYCL_END;

    LID *= 2;

    SYCL_IF(GID < global_size) {
      input[GID] += localBlock[LID];
      input[GID + 1] += localBlock[LID + 1];
    }
    SYCL_END;

    index.barrier(access::fence_space::local_space);

    uint1 last_sum_id = GID + local_size - 1;
    SYCL_IF(LID == 0 && last_sum_id < global_size) {
      // Write last sum into auxiliary array
      higher_level[GID / local_size] = input[last_sum_id];
    }
    SYCL_END;
  }
};

template <typename T>
struct prefix_sum_join_kernel {
 protected:
  using mode = cl::sycl::access::mode;
  using target = cl::sycl::access::target;
  using buffer = cl::sycl::buffer<T>;

  // Global memory
  cl::sycl::accessor<T> data;
  cl::sycl::accessor<T, 1, mode::read> higher_level;

 public:
  prefix_sum_join_kernel(cl::sycl::handler& cgh, buffer& data,
                         buffer& higher_level)
      : data(data.template get_access<mode::read_write>(cgh)),
        higher_level(higher_level.template get_access<mode::read>(cgh)) {}

  void operator()(cl::sycl::nd_item<1> index) {
    using namespace cl::sycl;
    int1 GID = 2 * index.get_global(0);
    int1 local_size = 2 * index.get_local_range()[0];
    SYCL_IF(GID >= local_size) {
      int1 higher_id = GID / local_size - 1;
      data[GID] += higher_level[higher_id];
      data[GID + 1] += higher_level[higher_id];
    }
    SYCL_END;
  }
};

template <typename type>
void prefix_sum_recursion(cl::sycl::queue& myQueue,
                          cl::sycl::vector_class<cl::sycl::buffer<type>>& data,
                          cl::sycl::vector_class<size_t>& sizes, size_t level,
                          size_t number_levels, size_t group_size) {
  using namespace cl::sycl;

  myQueue.submit([&](handler& cgh) {
    cgh.parallel_for(nd_range<1>(sizes[level] / 2, group_size),
                     prefix_sum_kernel<type>(cgh, data[level], data[level + 1],
                                             sizes[level], 2 * group_size));
  });

  if (level + 2 >= number_levels) {
    return;
  }

  prefix_sum_recursion(myQueue, data, sizes, level + 1, number_levels,
                       group_size);

  myQueue.submit([&](handler& cgh) {
    cgh.parallel_for(
        nd_range<1>(sizes[level] / 2, group_size),
        prefix_sum_join_kernel<type>(cgh, data[level], data[level + 1]));
  });
}

int main() {
  using namespace cl::sycl;

  {
    queue myQueue;

    const auto group_size =
        myQueue.get_device().get_info<info::device::max_work_group_size>();
    const auto size = group_size * 8;
    using type = float;

    // Calculate required number of buffer levels
    size_t N = size;
    size_t number_levels = 1;
    while (N > 1) {
      N /= 2 * group_size;
      ++number_levels;
    }

    // Create buffers
    vector_class<buffer<type>> data;
    vector_class<size_t> sizes;
    data.reserve(number_levels);
    sizes.reserve(number_levels);
    N = size;
    for (size_t i = 0; i < number_levels; ++i) {
      sizes.push_back(N);
      data.emplace_back(N);

      N = std::max(static_cast<size_t>(1), N / (2 * group_size));
      N += N % 2;  // Needs to be divisible by 2
    }

    // Init
    myQueue.submit([&](handler& cgh) {
      auto d = data[0].get_access<access::mode::write>(cgh);

      cgh.parallel_for<class init>(range<1>(size),
                                   [=](id<1> index) { d[index] = 1; });
    });

    prefix_sum_recursion(myQueue, data, sizes, 0, number_levels, group_size);

    debug() << "Done, checking results";
    return static_cast<int>(!check_sum(data[0]));
  }

  return 0;
}
