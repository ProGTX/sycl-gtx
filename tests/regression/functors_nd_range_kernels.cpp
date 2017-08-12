#include "../common.h"
#include <ctime>

// Passing functors as nd-range kernels

// Originally test5

using namespace cl::sycl;

class example_functor {
 public:
  using rw_acc_t =
      accessor<int, 1, access::mode::read_write, access::target::global_buffer>;

 private:
  rw_acc_t ptr;
  int random_num;

 public:
  example_functor(rw_acc_t p) : ptr(p) {
    random_num = std::rand() % (100 - 1) + 1;
  }

  void operator()(nd_item<1> item) {
    ptr[item.get_global()] = random_num;
  }

  int get_random() {
    return random_num;
  }
};

int main() {
  static const int group_size = 8;
  static const int size = group_size * 8;

  int data[size];
  srand(static_cast<unsigned int>(time(nullptr)));

  {
    queue myQueue;

    buffer<int> buf(data, size);
    int random_num = 0;

    myQueue.submit([&](handler& cgh) {
      auto ptr = buf.get_access<access::mode::read_write>(cgh);

      auto functor = example_functor(ptr);

      cgh.parallel_for(nd_range<1>(size, group_size), functor);

      random_num = functor.get_random();

      debug() << "-> Random number" << random_num;

    });

    auto hostPtr =
        buf.get_access<access::mode::read_write, access::target::host_buffer>();

    if (hostPtr[5] != random_num) {
      debug() << "The data retrieved from the device" << hostPtr[5]
              << "does not match the random number generated:" << random_num;
      return 1;
    }
  }

  return 0;
}
