#include "SYCL/device.h"
#include "SYCL/info.h"
#include "SYCL/platform.h"

using namespace cl::sycl;

device::device(cl_device_id device_id, device_selector* dev_sel)
    : device_id(device_id), platfrm(*dev_sel) {
  if (device_id == nullptr) {
    *this = dev_sel->select_device();
    this->device_id.release_one();
  }
}

device::device() : device(nullptr, detail::default_device_selector().get()) {}

device::device(cl_device_id device_id)
    : device(device_id, detail::default_device_selector().get()) {}

device::device(device_selector& dev_sel) : device(nullptr, &dev_sel) {}

cl_device_id device::get() const {
  return device_id.get();
}

bool device::is_host() const {
  // TODO(progtx):
  return false;
}

template <info::device_type type>
bool device::is_type() const {
  return get_info<info::device::device_type>() == type;
}

bool device::is_cpu() const {
  return is_type<info::device_type::cpu>();
}
bool device::is_gpu() const {
  return is_type<info::device_type::gpu>();
}
bool device::is_accelerator() const {
  return is_type<info::device_type::accelerator>();
}

platform device::get_platform() const {
  return platfrm;
}

vector_class<device> device::get_devices(info::device_type deviceType) {
  return detail::get_devices(static_cast<cl_device_type>(deviceType), nullptr);
}

bool device::has_extension(const string_class& extension_name) const {
  return detail::has_extension<info::device, info::device::extensions>(
      this, extension_name);
}

// TODO(progtx):
static vector_class<device> create_sub_devices(
    cl_device_id& did, const cl_device_partition_property* properties,
    int devices, unsigned int* num_devices) {
  cl_device_id* device_ids = new cl_device_id[devices];
  auto error_code =
      clCreateSubDevices(did, properties, devices, device_ids, num_devices);
  detail::error::report(error_code);
  auto device_vector =
      vector_class<device>(device_ids, device_ids + *num_devices);
  delete[] device_ids;
  return device_vector;
}

vector_class<device> detail::get_devices(cl_device_type device_type,
                                         cl_platform_id platform_id) {
  static const int MAX_DEVICES = 1024;
  cl_device_id device_ids[MAX_DEVICES];
  cl_uint num_devices;
  auto error_code = clGetDeviceIDs(platform_id, device_type, MAX_DEVICES,
                                   device_ids, &num_devices);
  detail::error::report(error_code);
  return vector_class<device>(device_ids, device_ids + num_devices);
}
