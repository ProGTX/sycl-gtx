#include "SYCL/context.h"
#include "SYCL/device.h"
#include "SYCL/platform.h"

using namespace cl::sycl;

// TODO(progtx): Master constructor
context::context(cl_context c, const async_handler& asyncHandler,
                 info::gl_context_interop interopFlag,
                 vector_class<device> deviceList, const platform* plt,
                 const device_selector& deviceSelector)
    : ctx(c), target_devices(deviceList), asyncHandler(asyncHandler) {
  if (c == nullptr) {
    cl_uint num_devices = static_cast<::cl_uint>(target_devices.size());

    if (num_devices == 0) {
      target_devices =
          device_selector::get_platform().get_devices(deviceSelector.type);
      num_devices = static_cast<::cl_uint>(target_devices.size());
    }

    vector_class<cl_device_id> devices;
    devices.reserve(num_devices);
    for (auto& device_ptr : target_devices) {
      devices.push_back(device_ptr.get());
    }

    ::cl_int error_code;
    c = clCreateContext(nullptr, num_devices, devices.data(), nullptr, nullptr,
                        &error_code);
    detail::error::report(error_code);
    ctx = c;
    ctx.release_one();
  }
}

context::context() : context(nullptr, detail::default_async_handler, false) {}

/**
 * Constructs a context object for SYCL host using an async_handler
 * for handling asynchronous errors.
 */
context::context(const async_handler& asyncHandler)
    : context(nullptr, asyncHandler, false) {}

/** Executes a retain on the cl_context */
context::context(cl_context clContext, const async_handler& asyncHandler)
    : context(clContext, asyncHandler, false) {}

context::context(const device_selector& deviceSelector,
                 info::gl_context_interop interopFlag,
                 const async_handler& asyncHandler)
    : context(nullptr, asyncHandler, interopFlag, {}, nullptr, deviceSelector) {
}

context::context(const device& dev, info::gl_context_interop interopFlag,
                 const async_handler& asyncHandler)
    : context(nullptr, asyncHandler, interopFlag, {dev}) {}

context::context(const platform& plt, info::gl_context_interop interopFlag,
                 const async_handler& asyncHandler)
    : context(nullptr, asyncHandler, interopFlag, {}, &plt) {}

context::context(vector_class<device> deviceList,
                 info::gl_context_interop interopFlag,
                 const async_handler& asyncHandler)
    : context(nullptr, asyncHandler, interopFlag, deviceList) {}

// TODO(progtx): Retain
cl_context context::get() const {
  return ctx.get();
}

vector_class<device> context::get_devices() const {
  return detail::transform_vector<device>(get_info<info::context::devices>());
}
