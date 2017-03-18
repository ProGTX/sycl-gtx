#include "../common.h"

#include <vector>

// Example from
// http://www.codeplay.com/portal/sycl-tutorial-1-the-vector-addition
// (slightly modified)

// Originally test1

#define TOL (0.001)    // Tolerance used in floating point comparisons
#define LENGTH (1024)  // Length of vectors a, b and c

int main() {
  using namespace cl::sycl;

  std::vector<int> h_a(LENGTH);              // a vector
  std::vector<int> h_b(LENGTH);              // b vector
  std::vector<int> h_c(LENGTH);              // c vector
  std::vector<int> h_r(LENGTH, 0xdeadbeef);  // d vector (result)

  // Fill vectors a and b with random float values
  debug() << "Initializing buffers";
  int count = LENGTH;
  for (int i = 0; i < count; i++) {
    h_a[i] = static_cast<int>(rand() / static_cast<float>(RAND_MAX));
    h_b[i] = static_cast<int>(rand() / static_cast<float>(RAND_MAX));
    h_c[i] = static_cast<int>(rand() / static_cast<float>(RAND_MAX));
  }

  {
    // Device buffers
    buffer<int> d_a(h_a);
    buffer<int> d_b(h_b);
    buffer<int> d_c(h_c);
    buffer<int> d_r(h_r);
    queue myQueue;
    debug() << "Submitting work";
    myQueue.submit([&](handler& cgh) {
      // Data accessors
      auto a = d_a.get_access<access::mode::read>(cgh);
      auto b = d_b.get_access<access::mode::read>(cgh);
      auto c = d_c.get_access<access::mode::read>(cgh);
      auto r = d_r.get_access<access::mode::write>(cgh);

      // Kernel
      cgh.parallel_for<class addition>(
          range<1>(count), [=](id<> i) { r[i] = a[i] + b[i] + c[i]; });
    });
  }

  debug() << "Done, checking results";
  int correct = 0;
  float tmp;
  for (int i = 0; i < count; i++) {
    tmp = static_cast<float>(h_a[i] + h_b[i] +
                             h_c[i]);  // assign element i of a + b + c to tmp
    tmp -= h_r[i];  // compute deviation of expected and output result
    if (tmp * tmp < TOL * TOL) {  // correct if square deviation
                                  // is less than tolerance squared
      correct++;
    } else {
      debug() << h_r[i] << "=\t" << h_a[i] << "+\t" << h_b[i] << "+\t"
              << h_c[i];
    }
  }

  // Summarize results
  debug() << "R = A+B+C:" << correct << "out of" << count
          << "results were correct.";

  return static_cast<int>(correct != count);
}
