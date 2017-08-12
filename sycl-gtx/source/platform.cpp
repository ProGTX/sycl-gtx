#include "SYCL/platform.h"
#include "SYCL/device.h"
#include "SYCL/info.h"

#include "SYCL/detail/debug.h"
#include <utility>

using namespace cl::sycl;

vector_class<platform> platform::platforms;

platform::platform(cl_platform_id platform_id, device_selector& dev_selector)
    : platform_id(platform_id) {}

platform::platform() : platform(nullptr) {}
platform::platform(cl_platform_id platform_id)
    : platform(platform_id, *detail::default_device_selector()) {}
platform::platform(device_selector& dev_selector)
    : platform(nullptr, dev_selector) {}

cl_platform_id platform::get() const {
  return platform_id.get();
}

vector_class<platform> platform::get_platforms() {
  if (platforms.empty()) {
    // TODO(progtx): Thread safe
    static const int MAX_PLATFORMS = 1024;
    cl_platform_id platform_ids[MAX_PLATFORMS];
    cl_uint num_platforms;
    auto error_code =
        clGetPlatformIDs(MAX_PLATFORMS, platform_ids, &num_platforms);
    detail::error::report(error_code);
    platforms =
        vector_class<platform>(platform_ids, platform_ids + num_platforms);
  }
  return platforms;
}

vector_class<device> platform::get_devices(
    info::device_type device_type) const {
  return detail::get_devices(static_cast<cl_device_type>(device_type),
                             platform_id.get());
}

// TODO(progtx): Check if SYCL running in Host Mode
bool platform::is_host() const {
  DSELF() << "not implemented";
  return false;
}

bool platform::has_extension(string_class extension_name) const {
  return detail::has_extension<info::platform, info::platform::extensions>(
      this, extension_name);
}
