#pragma once

#include "SYCL/detail/common.h"
#include "SYCL/info.h"

namespace cl {
namespace sycl {

// Forward declaration
template <typename EnumClass, EnumClass Value>
struct param_traits;

// Not in the specification, but is a nice addition
template <typename EnumClass, EnumClass Value>
using param_traits_t = typename param_traits<EnumClass, Value>::type;

namespace detail {

template <typename EnumClass, EnumClass Value, typename ReturnType,
          typename CLType>
struct param_traits_helper {
  using type = ReturnType;
  using cl_flag_type = CLType;
};

template <typename ReturnType>
struct traits_buffer_default {
  static const ::size_t size = 1;
};
template <typename Contained>
struct traits_buffer_default<vector_class<Contained>> {
  static const ::size_t size = 1024;
};
template <>
struct traits_buffer_default<string_class> {
  static const ::size_t size = 8192;
};

template <typename Contained_t, ::size_t BufferSize_v,
          class Container_t = vector_class<Contained_t>>
struct traits_helper {
  using Container = Container_t;
  using Contained = Contained_t;
  static const int BufferSizeConstant = BufferSize_v;
  static const ::size_t type_size = sizeof(Contained);
};

template <typename Contained_t,
          ::size_t BufferSize_v =
              traits_buffer_default<vector_class<Contained_t>>::size>
struct traits : traits_helper<Contained_t, BufferSize_v> {};

template <::size_t BufferSize_v>
struct traits<string_class, BufferSize_v>
    : traits_helper<char, BufferSize_v, string_class> {};

template <typename cl_input_t>
using opencl_info_f = ::cl_int(CL_API_CALL*)(cl_input_t, cl_uint, ::size_t,
                                             void*, ::size_t*);

template <typename cl_input_t, opencl_info_f<cl_input_t> F>
struct info_function_helper {
  template <class... Args>
  static ::cl_int get(Args... args) {
    return F(args...);
  }
};

template <typename EnumClass>
struct info_function;
template <>
struct info_function<info::detail::buffer>
    : info_function_helper<cl_mem, clGetMemObjectInfo> {};
template <>
struct info_function<info::context>
    : info_function_helper<cl_context, clGetContextInfo> {};
template <>
struct info_function<info::device>
    : info_function_helper<cl_device_id, clGetDeviceInfo> {};
template <>
struct info_function<info::event>
    : info_function_helper<cl_event, clGetEventInfo> {};
template <>
struct info_function<info::event_profiling>
    : info_function_helper<cl_event, clGetEventProfilingInfo> {};
template <>
struct info_function<info::kernel>
    : info_function_helper<cl_kernel, clGetKernelInfo> {};
template <>
struct info_function<info::platform>
    : info_function_helper<cl_platform_id, clGetPlatformInfo> {};
template <>
struct info_function<info::program>
    : info_function_helper<cl_program, clGetProgramInfo> {};
template <>
struct info_function<info::queue>
    : info_function_helper<cl_command_queue, clGetCommandQueueInfo> {};

template <bool IsSingleValue>
struct trait_return;
template <>
struct trait_return<false> {
  template <class Contained>
  static Contained* get(Contained* value) {
    return value;
  }
};
template <>
struct trait_return<true> {
  template <class Contained>
  static Contained get(Contained* value) {
    return value[0];
  }
};

}  // namespace detail

#define SYCL_ADD_TRAIT(EnumClass, Value, ReturnType, CLType) \
  template <>                                                \
  struct param_traits<EnumClass, Value>                      \
      : detail::param_traits_helper<EnumClass, Value, ReturnType, CLType> {};

/**
 * 3.3.3.2 Context information descriptors
 *
 * https://www.khronos.org/registry/cl/sdk/1.2/docs/man/xhtml/clGetContextInfo.html
 */
#define SYCL_ADD_CONTEXT_TRAIT(Value, ReturnType) \
  SYCL_ADD_TRAIT(info::context, Value, ReturnType, cl_context_info)

SYCL_ADD_CONTEXT_TRAIT(info::context::reference_count, cl_uint)
SYCL_ADD_CONTEXT_TRAIT(info::context::num_devices, cl_uint)
SYCL_ADD_CONTEXT_TRAIT(info::context::devices, vector_class<cl_device_id>)
SYCL_ADD_CONTEXT_TRAIT(info::context::gl_interop, info::gl_context_interop)

#undef SYCL_ADD_CONTEXT_TRAIT

/**
 * 3.3.2.1 Platform information descriptors
 *
 * https://www.khronos.org/registry/cl/sdk/1.2/docs/man/xhtml/clGetPlatformInfo.html
 */
#define SYCL_ADD_PLATFORM_TRAIT(Value) \
  SYCL_ADD_TRAIT(info::platform, Value, string_class, cl_platform_info)

SYCL_ADD_PLATFORM_TRAIT(info::platform::profile)
SYCL_ADD_PLATFORM_TRAIT(info::platform::version)
SYCL_ADD_PLATFORM_TRAIT(info::platform::name)
SYCL_ADD_PLATFORM_TRAIT(info::platform::vendor)
SYCL_ADD_PLATFORM_TRAIT(info::platform::extensions)

#undef SYCL_ADD_PLATFORM_TRAIT

/**
 * 3.3.4.2 Device information descriptors*
 * https://www.khronos.org/registry/cl/sdk/1.2/docs/man/xhtml/clGetDeviceInfo.html
 */
#define SYCL_ADD_DEVICE_TRAIT(Value, ReturnType) \
  SYCL_ADD_TRAIT(info::device, Value, ReturnType, cl_device_info)

// Forward declaration
template <int dimensions>
struct id;

SYCL_ADD_DEVICE_TRAIT(info::device::device_type, info::device_type)
SYCL_ADD_DEVICE_TRAIT(info::device::vendor_id, cl_uint)

SYCL_ADD_DEVICE_TRAIT(info::device::max_compute_units, cl_uint)
SYCL_ADD_DEVICE_TRAIT(info::device::max_work_item_dimensions, cl_uint)
SYCL_ADD_DEVICE_TRAIT(info::device::max_work_item_sizes, id<3>)
SYCL_ADD_DEVICE_TRAIT(info::device::max_work_group_size, ::size_t)

SYCL_ADD_DEVICE_TRAIT(info::device::preferred_vector_width_char, cl_uint)
SYCL_ADD_DEVICE_TRAIT(info::device::preferred_vector_width_short, cl_uint)
SYCL_ADD_DEVICE_TRAIT(info::device::preferred_vector_width_int, cl_uint)
SYCL_ADD_DEVICE_TRAIT(info::device::preferred_vector_width_long_long, cl_uint)
SYCL_ADD_DEVICE_TRAIT(info::device::preferred_vector_width_float, cl_uint)
SYCL_ADD_DEVICE_TRAIT(info::device::preferred_vector_width_double, cl_uint)
SYCL_ADD_DEVICE_TRAIT(info::device::preferred_vector_width_half, cl_uint)

SYCL_ADD_DEVICE_TRAIT(info::device::native_vector_width_char, cl_uint)
SYCL_ADD_DEVICE_TRAIT(info::device::native_vector_width_short, cl_uint)
SYCL_ADD_DEVICE_TRAIT(info::device::native_vector_width_int, cl_uint)
SYCL_ADD_DEVICE_TRAIT(info::device::native_vector_width_long_long, cl_uint)
SYCL_ADD_DEVICE_TRAIT(info::device::native_vector_width_float, cl_uint)
SYCL_ADD_DEVICE_TRAIT(info::device::native_vector_width_double, cl_uint)
SYCL_ADD_DEVICE_TRAIT(info::device::native_vector_width_half, cl_uint)

SYCL_ADD_DEVICE_TRAIT(info::device::max_clock_frequency, cl_uint)
SYCL_ADD_DEVICE_TRAIT(info::device::address_bits, cl_uint)
SYCL_ADD_DEVICE_TRAIT(info::device::max_mem_alloc_size, cl_ulong)

SYCL_ADD_DEVICE_TRAIT(info::device::image_support, cl_bool)
SYCL_ADD_DEVICE_TRAIT(info::device::max_read_image_args, cl_uint)
SYCL_ADD_DEVICE_TRAIT(info::device::max_write_image_args, cl_uint)
SYCL_ADD_DEVICE_TRAIT(info::device::image2d_max_height, ::size_t)
SYCL_ADD_DEVICE_TRAIT(info::device::image2d_max_width, ::size_t)
SYCL_ADD_DEVICE_TRAIT(info::device::image3d_max_height, ::size_t)
SYCL_ADD_DEVICE_TRAIT(info::device::image3d_max_width, ::size_t)
SYCL_ADD_DEVICE_TRAIT(info::device::image3d_max_depth, ::size_t)
SYCL_ADD_DEVICE_TRAIT(info::device::image_max_buffer_size, ::size_t)
SYCL_ADD_DEVICE_TRAIT(info::device::image_max_array_size, ::size_t)

SYCL_ADD_DEVICE_TRAIT(info::device::max_samplers, cl_uint)
SYCL_ADD_DEVICE_TRAIT(info::device::max_parameter_size, ::size_t)
SYCL_ADD_DEVICE_TRAIT(info::device::mem_base_addr_align, cl_uint)

SYCL_ADD_DEVICE_TRAIT(info::device::single_fp_config, info::device_fp_config)
SYCL_ADD_DEVICE_TRAIT(info::device::double_fp_config, info::device_fp_config)

SYCL_ADD_DEVICE_TRAIT(info::device::global_mem_cache_type,
                      info::global_mem_cache_type)
SYCL_ADD_DEVICE_TRAIT(info::device::global_mem_cache_line_size, cl_uint)
SYCL_ADD_DEVICE_TRAIT(info::device::global_mem_cache_size, cl_ulong)
SYCL_ADD_DEVICE_TRAIT(info::device::global_mem_size, cl_ulong)
SYCL_ADD_DEVICE_TRAIT(info::device::max_constant_buffer_size, cl_ulong)
SYCL_ADD_DEVICE_TRAIT(info::device::max_constant_args, cl_uint)
SYCL_ADD_DEVICE_TRAIT(info::device::local_mem_type, info::local_mem_type)
SYCL_ADD_DEVICE_TRAIT(info::device::local_mem_size, cl_ulong)

SYCL_ADD_DEVICE_TRAIT(info::device::error_correction_support, cl_bool)
SYCL_ADD_DEVICE_TRAIT(info::device::host_unified_memory, cl_bool)
SYCL_ADD_DEVICE_TRAIT(info::device::profiling_timer_resolution, ::size_t)
SYCL_ADD_DEVICE_TRAIT(info::device::endian_little, cl_bool)
SYCL_ADD_DEVICE_TRAIT(info::device::is_available, cl_bool)
SYCL_ADD_DEVICE_TRAIT(info::device::is_compiler_available, cl_bool)
SYCL_ADD_DEVICE_TRAIT(info::device::is_linker_available, cl_bool)

SYCL_ADD_DEVICE_TRAIT(info::device::execution_capabilities,
                      info::device_exec_capabilities)
SYCL_ADD_DEVICE_TRAIT(info::device::queue_properties,
                      info::device_queue_properties)
SYCL_ADD_DEVICE_TRAIT(info::device::built_in_kernels, string_class)
SYCL_ADD_DEVICE_TRAIT(info::device::platform, cl_platform_id)
SYCL_ADD_DEVICE_TRAIT(info::device::name, string_class)
SYCL_ADD_DEVICE_TRAIT(info::device::vendor, string_class)
SYCL_ADD_DEVICE_TRAIT(info::device::driver_version, string_class)
SYCL_ADD_DEVICE_TRAIT(info::device::profile, string_class)
SYCL_ADD_DEVICE_TRAIT(info::device::device_version, string_class)
SYCL_ADD_DEVICE_TRAIT(info::device::opencl_version, string_class)
SYCL_ADD_DEVICE_TRAIT(info::device::extensions, string_class)

SYCL_ADD_DEVICE_TRAIT(info::device::printf_buffer_size, ::size_t)
SYCL_ADD_DEVICE_TRAIT(info::device::preferred_interop_user_sync, cl_bool)
SYCL_ADD_DEVICE_TRAIT(info::device::parent_device, cl_device_id)
SYCL_ADD_DEVICE_TRAIT(info::device::partition_max_sub_devices, cl_uint)
SYCL_ADD_DEVICE_TRAIT(info::device::partition_properties,
                      vector_class<info::device_partition_property>)
SYCL_ADD_DEVICE_TRAIT(info::device::partition_affinity_domain,
                      info::device_affinity_domain)
SYCL_ADD_DEVICE_TRAIT(
    info::device::partition_type,
    vector_class<info::device_partition_type>)  // TODO(progtx):
SYCL_ADD_DEVICE_TRAIT(info::device::reference_count, cl_uint)

#undef SYCL_ADD_DEVICE_TRAIT

/**
 * 3.3.5.2 Queue information descriptors*
 * https://www.khronos.org/registry/cl/sdk/1.2/docs/man/xhtml/clGetCommandQueueInfo.html
 */
#define SYCL_ADD_QUEUE_TRAIT(Value, ReturnType) \
  SYCL_ADD_TRAIT(info::queue, Value, ReturnType, cl_command_queue_info)

SYCL_ADD_QUEUE_TRAIT(info::queue::context, cl_context)
SYCL_ADD_QUEUE_TRAIT(info::queue::device, cl_device_id)
SYCL_ADD_QUEUE_TRAIT(info::queue::reference_count, cl_uint)
SYCL_ADD_QUEUE_TRAIT(info::queue::properties, info::queue_profiling)

#undef SYCL_ADD_QUEUE_TRAIT

/**
 * https://www.khronos.org/registry/cl/sdk/1.2/docs/man/xhtml/clGetMemObjectInfo.html
 */
#define SYCL_ADD_BUFFER_TRAIT(Value, ReturnType) \
  SYCL_ADD_TRAIT(info::detail::buffer, Value, ReturnType, cl_mem_info)

SYCL_ADD_BUFFER_TRAIT(info::detail::buffer::type, cl_mem_object_type)
SYCL_ADD_BUFFER_TRAIT(info::detail::buffer::flags, cl_mem_flags)
SYCL_ADD_BUFFER_TRAIT(info::detail::buffer::size, ::size_t)
SYCL_ADD_BUFFER_TRAIT(info::detail::buffer::host_pointer, void*)
SYCL_ADD_BUFFER_TRAIT(info::detail::buffer::map_count, cl_uint)
SYCL_ADD_BUFFER_TRAIT(info::detail::buffer::reference_count, cl_uint)
SYCL_ADD_BUFFER_TRAIT(info::detail::buffer::context, cl_context)
SYCL_ADD_BUFFER_TRAIT(info::detail::buffer::associated_memory_object, cl_mem)
SYCL_ADD_BUFFER_TRAIT(info::detail::buffer::offset, ::size_t)

#undef SYCL_ADD_BUFFER_TRAIT

/**
 * Table 3.62: Kernel class information descriptors.
 *
 * https://www.khronos.org/registry/cl/sdk/1.2/docs/man/xhtml/clGetKernelInfo.html
 */
#define SYCL_ADD_KERNEL_TRAIT(Value, ReturnType) \
  SYCL_ADD_TRAIT(info::kernel, Value, ReturnType, cl_kernel_info)

SYCL_ADD_KERNEL_TRAIT(info::kernel::function_name, string_class)
SYCL_ADD_KERNEL_TRAIT(info::kernel::num_args, cl_uint)
SYCL_ADD_KERNEL_TRAIT(info::kernel::reference_count, cl_uint)
SYCL_ADD_KERNEL_TRAIT(info::kernel::attributes, string_class)

// Not part of the SYCL specification
SYCL_ADD_KERNEL_TRAIT(info::kernel::context, cl_context)
SYCL_ADD_KERNEL_TRAIT(info::kernel::program, cl_program)

#undef SYCL_ADD_KERNEL_TRAIT

/**
 * Table 3.65: Program class information descriptors
 *
 * https://www.khronos.org/registry/cl/sdk/1.2/docs/man/xhtml/clGetProgramInfo.html
 */
#define SYCL_ADD_PROGRAM_TRAIT(Value, ReturnType) \
  SYCL_ADD_TRAIT(info::program, Value, ReturnType, cl_program_info)

SYCL_ADD_PROGRAM_TRAIT(info::program::reference_count, cl_uint)
SYCL_ADD_PROGRAM_TRAIT(info::program::context, cl_context)
SYCL_ADD_PROGRAM_TRAIT(info::program::devices, vector_class<cl_device_id>)

// Not part of the SYCL specification
SYCL_ADD_PROGRAM_TRAIT(info::program::num_devices, cl_uint)
SYCL_ADD_PROGRAM_TRAIT(info::program::source, string_class)
SYCL_ADD_PROGRAM_TRAIT(info::program::binary_sizes, vector_class<::size_t>)
SYCL_ADD_PROGRAM_TRAIT(info::program::binaries,
                       vector_class<vector_class<unsigned char>>)
SYCL_ADD_PROGRAM_TRAIT(info::program::num_kernels, ::size_t)
SYCL_ADD_PROGRAM_TRAIT(info::program::kernel_names, string_class)

#undef SYCL_ADD_PROGRAM_TRAIT

/**
 * 3.3.6.1 Event information and profiling descriptors
 *
 * https://www.khronos.org/registry/cl/sdk/1.2/docs/man/xhtml/clGetEventInfo.html
 */
#define SYCL_ADD_EVENT_TRAIT(Value, ReturnType) \
  SYCL_ADD_TRAIT(info::event, Value, ReturnType, cl_event_info)

SYCL_ADD_EVENT_TRAIT(info::event::command_type, cl_command_type)
SYCL_ADD_EVENT_TRAIT(info::event::command_execution_status, ::cl_int)
SYCL_ADD_EVENT_TRAIT(info::event::reference_count, ::cl_int)

// Not part of the SYCL specification
SYCL_ADD_EVENT_TRAIT(info::event::command_queue, cl_command_queue)
SYCL_ADD_EVENT_TRAIT(info::event::context, cl_context)

#undef SYCL_ADD_EVENT_TRAIT

/**
 * https://www.khronos.org/registry/cl/sdk/1.2/docs/man/xhtml/clGetEventProfilingInfo.html
 */
#define SYCL_ADD_EVENT_PROFILING_TRAIT(Value, ReturnType) \
  SYCL_ADD_TRAIT(info::event_profiling, Value, ReturnType, cl_profiling_info)

SYCL_ADD_EVENT_PROFILING_TRAIT(info::event_profiling::command_queued, cl_ulong)
SYCL_ADD_EVENT_PROFILING_TRAIT(info::event_profiling::command_submit, cl_ulong)
SYCL_ADD_EVENT_PROFILING_TRAIT(info::event_profiling::command_start, cl_ulong)
SYCL_ADD_EVENT_PROFILING_TRAIT(info::event_profiling::command_end, cl_ulong)

#undef SYCL_ADD_EVENT_PROFILING_TRAIT

#undef SYCL_ADD_TRAIT

namespace detail {

template <class Contained_t, class EnumClass, EnumClass param,
          ::size_t BufferSize_v = traits<Contained_t>::BufferSizeConstant>
struct array_traits : traits<Contained_t, BufferSize_v> {
  using Base = array_traits<Contained_t, EnumClass, param, BufferSize_v>;
  using RealBase = traits<Contained_t, BufferSize_v>;
  using Contained = typename RealBase::Contained;
  using return_t =
      typename std::conditional<BufferSize_v == 1, Contained, Contained*>::type;
  Contained param_value[RealBase::BufferSizeConstant];
  ::size_t actual_size = 0;

  template <typename cl_input_t>
  return_t get(cl_input_t data_ptr) {
    auto error_code = info_function<EnumClass>::get(
        data_ptr,
        static_cast<typename param_traits<EnumClass, param>::cl_flag_type>(
            param),
        RealBase::BufferSizeConstant * RealBase::type_size, param_value,
        &actual_size);
    error::report(error_code);
    return trait_return<BufferSize_v == 1>::get(param_value);
  }
};

/** Meant for scalar and string cases */
template <class EnumClass, EnumClass param,
          ::size_t BufferSize_v =
              traits<param_traits_t<EnumClass, param>>::BufferSizeConstant>
struct non_vector_traits : array_traits<param_traits_t<EnumClass, param>,
                                        EnumClass, param, BufferSize_v> {};

}  // namespace detail

}  // namespace sycl
}  // namespace cl
