#include "SYCL/device_selector.h"
#include "SYCL/detail/debug.h"
#include "SYCL/device.h"
#include "SYCL/platform.h"

using namespace cl::sycl;

device device_selector::select_device(vector_class<device> devices) const {
  int best_id = -1;
  int best_score = -1;
  int i = 0;

  for (auto& dev : devices) {
    int score = operator()(dev);
    if (score > best_score) {
      best_id = i;
      best_score = score;
    }
    ++i;
  }

  // Devices with a negative score will never be chosen.
  if (best_score < 0) {
    // TODO(progtx): The "default" device constructed corresponds to the host.
    // This is also the device that the system will "fall-back" to,
    // if there are no existing or valid OpenCL devices associated with the
    // system.
    debug::warning(__func__) << "does not support a default device yet";
    throw std::exception();
  } else {
    return devices[best_id];
  }
}

platform device_selector::get_platform() {
  auto platforms = platform::get_platforms();
  // TODO(progtx): Platform selection
  return std::move(platforms[0]);
}

device device_selector::select_device() const {
  return select_device(get_platform().get_devices(type));
}

int default_selector::operator()(const device& dev) const {
  return 0;
}

int gpu_selector::operator()(const device& dev) const {
  return 0;
}

int cpu_selector::operator()(const device& dev) const {
  return 0;
}

int host_selector::operator()(const device& dev) const {
  return 0;
}
