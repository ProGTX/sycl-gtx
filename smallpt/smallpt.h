#pragma once

#include <CL/sycl.hpp>

#include <chrono>
#include <iostream>
#include <memory>
#include <string>
#include <vector>

#ifndef float_type
#define float_type double
#endif

#include "classes.h"
#include "win.h"

#ifndef modify_sample_rate
#define modify_sample_rate 1
#endif

using Vec = Vec_detail<float_type>;
using Ray = Ray_detail<float_type>;
using Sphere = Sphere_detail<float_type, modify_sample_rate>;

inline float_type clamp(float_type x) {
  return x < 0 ? 0 : x > 1 ? 1 : x;
}
inline int toInt(float_type x);

static void to_file(int w, int h, Vec* c, std::string filename) {
  FILE* f = fopen(filename.c_str(), "w");  // Write image to PPM file.
  fprintf(                                 // NOLINT
      f, "P3\n%d %d\n%d\n", w, h, 255);
  for (int i = 0; i < w * h; i++) {
    fprintf(  // NOLINT
        f, "%d %d %d\n", toInt(c[i].x), toInt(c[i].y), toInt(c[i].z));
  }
  fclose(f);
}

using time_point = std::chrono::high_resolution_clock::time_point;

static auto now = []() { return std::chrono::high_resolution_clock::now(); };
static auto duration = [](time_point before) {
  static const float to_seconds = 1e-6f;
  return std::chrono::duration_cast<std::chrono::microseconds>(now() - before)
             .count() *
         to_seconds;
};

struct testInfo {
  using function_ptr = void (*)(void*, int, int, int, Ray, Vec, Vec, Vec, Vec*);
  std::string name;
  function_ptr test;
  std::shared_ptr<cl::sycl::device> dev;
  float lastTime = 0;

  testInfo(std::string name, function_ptr test,
           std::shared_ptr<cl::sycl::device> dev = nullptr)
      : name(name), test(test), dev(dev) {}

  testInfo(const testInfo&) = delete;
  testInfo(testInfo&& move) noexcept
      : name(std::move(move.name)), test(move.test), dev(std::move(move.dev)) {}

  testInfo& operator=(const testInfo&) = delete;
  testInfo& operator=(testInfo&& move) noexcept = delete;
  ~testInfo() = default;

  bool isOpenCL() {
    return dev != nullptr;
  }
};

static decltype(now())& startTime() {
  static decltype(now()) s(now());
  return s;
}

static Ray& cam() {
  static Ray c(Vec(50, 52, 295.6),
               Vec(0, -0.042612, -1).norm());  // cam pos, dir
  return c;
}

static std::string& imagePrefix() {
  static std::string ip;
  return ip;
}

static bool tester(std::vector<testInfo>& tests, int maxMinutes, int w, int h,
                   int samples, Vec& cx, Vec& cy, int iterations, int from,
                   int to) {
  using namespace std;

  if (to - from <= 0) {
    cout << "no tests" << endl;
    return false;
  }
  bool isOpenCLAvailable = tests[to - 1].isOpenCL();

  cout << "samples per pixel: " << samples << endl;

  Vec r;
  vector<Vec> empty_vectors(w * h, 0);
  vector<Vec> vectors;
  float time;

  const float perTestLimit = 40;
  const float globalLimit = 60.f * maxMinutes;
  float totalTime = 0;

  for (int ti = from; ti < to; ++ti) {
    auto& t = tests[ti];

    // Quality of Service
    // Prevents the test from taking too long,
    // but also allows it to use up as much time as possible
    // OpenCL tests are preferred
    bool overHalf = 2 * totalTime > globalLimit;
    if (t.lastTime > perTestLimit &&
        ((!isOpenCLAvailable && overHalf) ||
         (isOpenCLAvailable &&
          (!t.isOpenCL() || (t.isOpenCL() && overHalf))))) {
      continue;
    }

    cout << "Running test: " << t.name << endl;
    ns_random::reset();
#ifdef NDEBUG
    try {
#endif
      auto start = now();
      for (int i = 0; i < iterations; ++i) {
        vectors = empty_vectors;
        t.test(t.dev.get(), w, h, samples, cam(), cx, cy, r, vectors.data());
      }
      time = (duration(start) / static_cast<float>(iterations));

#ifdef NDEBUG
    } catch (cl::sycl::exception& e) {
      cerr << "SYCL error while testing: " << e.what() << endl;
      continue;
    } catch (std::exception& e) {
      cerr << "error while testing: " << e.what() << endl;
      continue;
    }
#endif

#ifndef NDEBUG
    to_file(w, h, vectors.data(), imagePrefix() + ' ' + t.name + ".ppm");
#endif

    cout << "time: " << time << endl;
    t.lastTime = time;
    totalTime = duration(startTime());
    cout << "total time: " << std::to_string(totalTime) << endl;
    if (totalTime > globalLimit) {
      cout << "exceeded " + std::to_string(static_cast<int>(globalLimit)) +
                  "s limit, stopping"
           << endl;
      return false;
    }
  }

  return true;
}

struct version {
  int major = 0;
  int minor = 0;

  version(int major, int minor) : major{major}, minor{minor} {}
  version(const std::string& v) {
    // https://www.khronos.org/registry/cl/sdk/1.2/docs/man/xhtml/clGetPlatformInfo.html
    using namespace std;
    std::string search("OpenCL");
    auto pos = v.find(search);
    if (pos != std::string::npos) {
      pos += search.length() + 1;  // Plus one for space
      try {
        major = static_cast<int>(v.at(pos)) - '0';
        minor = static_cast<int>(v.at(pos + 2)) - '0';
        ;  // Plus one for dot
      } catch (std::exception&) {
      }
    }
  }
};

template <class T>
void printInfo(std::string description, const T& data, int offset = 0) {
  std::string indent;
  for (int i = 0; i < offset; ++i) {
    indent += '\t';
  }
  std::cout << indent << description << ": " << data << std::endl;
}

static void displayDevice(const cl::sycl::device& d, int dNum,
                          std::string& name, version& deviceVersion,
                          int tabOffset = 2) {
  using namespace std;
  using namespace cl::sycl;
  cout << "\t-- OpenCL device " << dNum << ':' << endl;

  name = d.get_info<info::device::name>();
  auto deviceVersionString = d.get_info<info::device::device_version>();
  deviceVersion = deviceVersionString;

  printInfo("name", name, tabOffset);
  printInfo(
      "device_type",
      static_cast<cl_device_type>(d.get_info<info::device::device_type>()),
      tabOffset);
  printInfo("vendor", d.get_info<info::device::vendor>(), tabOffset);
  printInfo("device_version", deviceVersionString, tabOffset);
  printInfo("driver_version", d.get_info<info::device::driver_version>(),
            tabOffset);
#ifdef SYCL_GTX
  printInfo("opencl_version", d.get_info<info::device::opencl_version>(),
            tabOffset);
  printInfo("single_fp_config", d.get_info<info::device::single_fp_config>(),
            tabOffset);
  printInfo("double_fp_config", d.get_info<info::device::double_fp_config>(),
            tabOffset);
#endif
  printInfo("profile", d.get_info<info::device::profile>(), tabOffset);
  printInfo("error_correction_support",
            d.get_info<info::device::error_correction_support>(), tabOffset);
  printInfo("host_unified_memory",
            d.get_info<info::device::host_unified_memory>(), tabOffset);
  printInfo("max_clock_frequency",
            d.get_info<info::device::max_clock_frequency>(), tabOffset);
  printInfo("max_compute_units", d.get_info<info::device::max_compute_units>(),
            tabOffset);
  printInfo("max_work_item_dimensions",
            d.get_info<info::device::max_work_item_dimensions>(), tabOffset);
  printInfo("max_work_group_size",
            d.get_info<info::device::max_work_group_size>(), tabOffset);

  printInfo("address_bits", d.get_info<info::device::address_bits>(),
            tabOffset);
  printInfo("max_mem_alloc_size",
            d.get_info<info::device::max_mem_alloc_size>(), tabOffset);
  printInfo("global_mem_cache_line_size",
            d.get_info<info::device::global_mem_cache_line_size>(), tabOffset);
  printInfo("global_mem_cache_size",
            d.get_info<info::device::global_mem_cache_size>(), tabOffset);
  printInfo("global_mem_size", d.get_info<info::device::global_mem_size>(),
            tabOffset);
  printInfo("max_constant_buffer_size",
            d.get_info<info::device::max_constant_buffer_size>(), tabOffset);
  printInfo("max_constant_args", d.get_info<info::device::max_constant_args>(),
            tabOffset);
  printInfo("local_mem_size", d.get_info<info::device::local_mem_size>(),
            tabOffset);
  printInfo("extensions", d.get_info<info::device::extensions>(), tabOffset);
}

static void getDevices(std::vector<testInfo>& tests,
                       std::vector<testInfo::function_ptr> compute_sycl_ptrs) {
  using namespace std;

  try {
    using namespace cl::sycl;

    auto platforms = platform::get_platforms();
    std::vector<testInfo> testVector;

    version required(1, 2);

    int pNum = 0;
    for (auto& p : platforms) {
      cout << "- OpenCL platform " << pNum << ':' << endl;
      ++pNum;

      auto openclVersion = p.get_info<info::platform::version>();
      version platformVersion(openclVersion);

      printInfo("name", p.get_info<info::platform::name>(), 1);
      printInfo("vendor", p.get_info<info::platform::vendor>(), 1);
      printInfo("version", openclVersion, 1);
      printInfo("profile", p.get_info<info::platform::profile>(), 1);
      printInfo("extensions", p.get_info<info::platform::extensions>(), 1);

      auto devices = p.get_devices();
      int dNum = 0;

      for (auto& d : devices) {
        std::string name;
        version deviceVersion("");
        displayDevice(d, dNum, name, deviceVersion);

#ifndef SYCL_GTX
        // TODO: ComputeCpp returns wrong values
        deviceVersion = platformVersion;
#endif
        if (deviceVersion.major > required.major ||
            ((deviceVersion.major == required.major) &&
             (deviceVersion.minor >= required.minor))) {
#ifndef SYCL_GTX
#ifndef NDEBUG
          // There seem to be some problems with ComputeCpp and HD 4600
          if (name.find("HD Graphics 4600") == std::string::npos)
#endif
#endif
            testVector.emplace_back(
                name + ' ' + openclVersion, nullptr,
                std::shared_ptr<device>(new device(std::move(d))));
        }

        ++dNum;
      }
    }

    int i = 0;
    for (auto ptr : compute_sycl_ptrs) {
      ++i;
      for (auto& t : testVector) {
        tests.emplace_back(std::string("T") + std::to_string(i) + ' ' + t.name,
                           ptr, t.dev);
      }
    }
  } catch (cl::sycl::exception& e) {
    // TODO(progtx):
    cout << "OpenCL not available: " << e.what() << endl;
  }
}

static int mainTester(int argc, char* argv[], std::vector<testInfo>& tests,
                      std::string image_prefix) {
  using namespace std;

  cout << "smallpt SYCL tester" << endl;

  imagePrefix() = image_prefix;

  int w = 1024;
  int h = 768;
  Vec cx = Vec(w * .5135 / h);
  Vec cy = (cx % cam().d).norm() * .5135;
  auto numTests = tests.size();

  int from = 0;
  int to = numTests;
  int maxMinutes = 5;

  if (argc > 1) {
    maxMinutes = atoi(argv[1]);
    if (argc > 2) {
      from = atoi(argv[2]);
      if (argc > 3) {
        to = atoi(argv[3]);
      }
    }
  }

  cout << "Global time limit in minutes: " << maxMinutes << endl;
  cout << "Going through tests in range [" << from << ',' << to << ')' << endl;

  startTime();

  if (false) {
    from = 0;
    to = 1;
    auto prefix = imagePrefix();

    imagePrefix() = prefix + "_64";
    tester(tests, maxMinutes, w, h, 64, cx, cy, 1, from, to);
  } else {
    // Test suite
    int iterations = 1;
    bool canContinue;

    for (int samples = 4; samples < 10000; samples *= 2) {
      canContinue = tester(tests, maxMinutes, w, h, samples, cx, cy, iterations,
                           from, to);
      if (!canContinue) {
        break;
      }
    }
  }

  auto time = duration(startTime());
  cout << "total test suite duration: " << time << endl;
  // cout << "Press any key to exit" << endl;
  // cin.get();

  return 0;
}
