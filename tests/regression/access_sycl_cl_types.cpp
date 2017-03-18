#include "../common.h"

// Check access into SYCL CL types

// Originally test12

int main() {
  using namespace std;
  using namespace cl::sycl;

  auto size = 1;
  auto spheres = buffer<int8>(range<1>(size));

  ::cl_int8 testVector = {0, 1, 2, 3, 4, 5, 6, 7};

  {
    auto s = spheres.get_access<access::mode::discard_write,
                                access::target::host_buffer>();

    // See SphereSycl
    for (int i = 0; i < size; ++i) {
      auto& si = s[i];

      si.lo().x() = testVector.s[0] * (i + 1);
      si.lo().y() = testVector.s[1] * (i + 1);
      si.lo().z() = testVector.s[2] * (i + 1);
      si.lo().w() = testVector.s[3] * (i + 1);

      si.hi().x() = testVector.s[4] * (i + 1);
      si.hi().y() = testVector.s[5] * (i + 1);
      si.hi().z() = testVector.s[6] * (i + 1);
      si.hi().w() = testVector.s[7] * (i + 1);
    }
  }

  {
    auto s =
        spheres.get_access<access::mode::read, access::target::host_buffer>();

    auto compare = [](::cl_int expected, ::cl_int actual) {
#ifndef NDEBUG
      cout << expected << ", " << actual << endl;
#endif
      if (actual != expected) {
        cout << "Wrong output, expected " << expected << ", got " << actual
             << endl;
        return false;
      }
      return true;
    };

    // See SphereSycl
    for (int i = 0; i < size; ++i) {
      auto& si = s[i];

      if (!compare(testVector.s[0], si.lo().x())) {
        return 1;
      }
      if (!compare(testVector.s[1], si.lo().y())) {
        return 1;
      }
      if (!compare(testVector.s[2], si.lo().z())) {
        return 1;
      }
      if (!compare(testVector.s[3], si.lo().w())) {
        return 1;
      }

      if (!compare(testVector.s[4], si.hi().x())) {
        return 1;
      }
      if (!compare(testVector.s[5], si.hi().y())) {
        return 1;
      }
      if (!compare(testVector.s[6], si.hi().z())) {
        return 1;
      }
      if (!compare(testVector.s[7], si.hi().w())) {
        return 1;
      }
    }
  }

  return 0;
}
