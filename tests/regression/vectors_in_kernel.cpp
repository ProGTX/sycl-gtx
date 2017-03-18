#include "../common.h"

// Test vectors in kernel

// Originally test10

#include <sstream>

template <class T>
std::string to_string(const T& t) {
  std::stringstream s;
  s << '(' << t.x() << ',' << t.y() << ',' << t.z() << ')';
  return s.str();
}

int main() {
  using namespace cl::sycl;
  using namespace std;

  cpu_selector cpu;
  queue myQueue(cpu);

  const int size = 10;
  cl::sycl::cl_float3 testVector;
  testVector.x() = 1;
  testVector.y() = 2;
  testVector.z() = 3;
  buffer<float3> vectors(size);

  myQueue.submit([&](handler& cgh) {
    auto v = vectors.get_access<access::mode::discard_write>(cgh);

    cgh.parallel_for<class addition>(range<1>(size), [=](id<> i) {
      v[i] = float3(testVector.x(), testVector.y(), 0);
      v[i].z() = testVector.z();
    });
  });

  auto v =
      vectors.get_access<access::mode::read, access::target::host_buffer>();

  auto floatEqual = [](float& first, float& second) {
    static const double eps = 1e-5f;
    return first > second - eps && first < second + eps;
  };

  for (auto i = 0; i < size; ++i) {
    auto vi = v[i];
    if (!floatEqual(vi.x(), testVector.x()) ||
        !floatEqual(vi.y(), testVector.y()) ||
        !floatEqual(vi.z(), testVector.z())) {
      cout << i << " -> expected " << to_string(testVector) << ", got "
           << to_string(vi) << endl;
      return 1;
    }
  }

  return 0;
}
