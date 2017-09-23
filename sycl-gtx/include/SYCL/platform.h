#pragma once

// 3.3.2 Platform class

#include "SYCL/detail/common.h"
#include "SYCL/device_selector.h"
#include "SYCL/error_handler.h"
#include "SYCL/info.h"
#include "SYCL/param_traits.h"
#include "SYCL/refc.h"

namespace cl {
namespace sycl {

// Forward declaration
class device;

/** Constructors return errors via C++ exception class. */
class platform {
 private:
  detail::refc<cl_platform_id> platform_id;

  platform(cl_platform_id platform_id, device_selector& dev_selector);

  static vector_class<platform> platforms;

 public:
  /**
   * Default constructor for platform.
   * It constructs a platform object to encapsulate the device
   * returned by the default device selector
   */
  platform();

  /** Construct a platform object from an OpenCL platform id. */
  explicit platform(cl_platform_id platform_id);

  /**
   * Construct a platform object from the device
   * returned by a device selector of the user's choice.
   */
  explicit platform(device_selector& dev_selector);

  /** The OpenCL cl_platform_id or nullptr for SYCL host */
  cl_platform_id get() const;

  /** Returns all the available OpenCL platforms and the SYCL host platform */
  static vector_class<platform> get_platforms();

  /** Returns the devices available in this platform */
  vector_class<device> get_devices(
      info::device_type = info::device_type::all) const;

  /**
   * Returns the corresponding descriptor information
   * for all SYCL platforms (OpenCL and host)
   */
  template <info::platform param>
  typename param_traits<info::platform, param>::type get_info() const {
    // Small optimization, knowing the return type is always string_class
    return detail::non_vector_traits<
               info::platform, param,
               detail::traits_buffer_default<string_class>::size>()
        .get(platform_id.get());
  }

  /** True if the platform is host */
  bool is_host() const;

  /** Returns the available extensions for all SYCL platforms (OpenCL and host)
   */
  bool has_extension(string_class extension_name) const;
};

}  // namespace sycl
}  // namespace cl
