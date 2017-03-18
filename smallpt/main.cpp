
#define float_type double
#include "smallpt.h"
#include "sycl_gtx.h"

#include <cmath>

extern void compute_org(void*, int w, int h, int samps, Ray cam, Vec cx, Vec cy,
                        Vec r, Vec* c);
extern void compute_org_openmp(void*, int w, int h, int samps, Ray cam, Vec cx,
                               Vec cy, Vec r, Vec* c);
extern void compute_org_sp(void*, int w, int h, int samps, Ray cam, Vec cx,
                           Vec cy, Vec r, Vec* c);
extern void compute_org_sp_openmp(void*, int w, int h, int samps, Ray cam,
                                  Vec cx, Vec cy, Vec r, Vec* c);
extern void compute_sycl_gtx(void*, int w, int h, int samps, Ray cam, Vec cx,
                             Vec cy, Vec r, Vec* c);

inline int toInt(float_type x) {
  return std::lround(pow(clamp(x), 1 / 2.2) * 255);
}

int main(int argc, char* argv[]) {
  using namespace std;
  vector<testInfo> tests;

  // tests.emplace_back("org_single", compute_org_sp);
  // tests.emplace_back("openmp_single", compute_org_sp_openmp);
  tests.emplace_back("org", compute_org);
  tests.emplace_back("openmp", compute_org_openmp);

  getDevices(tests, {compute_sycl_gtx});

  return mainTester(argc, argv, tests, "sycl_gtx");
}
