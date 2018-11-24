#pragma once

// 3.4.2 Buffers

#include "SYCL/access.h"
#include "SYCL/buffer_base.h"
#include "SYCL/command_group.h"
#include "SYCL/detail/common.h"
#include "SYCL/detail/debug.h"
#include "SYCL/detail/synchronizer.h"
#include "SYCL/error_handler.h"
#include "SYCL/event.h"
#include "SYCL/info.h"
#include "SYCL/param_traits.h"
#include "SYCL/ranges.h"
#include "SYCL/refc.h"
#include <algorithm>

namespace cl {
namespace sycl {

// Forward declarations
template <typename, int, access::mode, access::target>
class accessor;
template <typename, int = 1>
struct buffer;
class handler;
class queue;

namespace detail {

// Forward declarations
template <typename, int, access::mode, access::target, typename>
class accessor_detail;
template <typename, int>
class accessor_buffer;
class command_group;

#undef SYCL_ADD_ACCESS_MODE_HELPER

template <typename DataType_t, int dimensions>
class buffer_detail : public buffer_base {
 public:
  using value_type = typename base_host_data<DataType_t>::type;
  using reference = value_type&;
  using const_reference = const value_type&;

 protected:
  using DataType = value_type;
  using ptr_t = shared_ptr_class<DataType>;

  range<dimensions> rang;
  ptr_t host_data;

  bool is_read_only = false;
  bool is_blocking = true;
  bool is_initialized = false;

  friend class accessor_base;
  friend class accessor_buffer<DataType_t, dimensions>;
  friend class kernel_ns::source;

  /** Associated host memory. */
  buffer_detail(value_type* host_data, range<dimensions> range,
                bool is_read_only, bool is_blocking = true)
      : host_data(ptr_t(host_data, [](value_type* ptr) {})),
        rang(range),
        is_read_only(is_read_only),
        is_blocking(is_blocking) {}

  buffer_detail(std::nullptr_t host_data, range<dimensions> range)
      : buffer_detail(nullptr, range, false) {}

 public:
  /**
   * Creates a new buffer with associated host memory.
   * The memory is owned by the runtime during the lifetime of the object.
   * Data is copied back to the host unless the user overrides the behavior
   * using the set_final_data method.
   * @param hostData points to the storage and values used by the buffer
   * @param range<dimensions> defines the size.
   */
  buffer_detail(DataType* hostData, range<dimensions> range)
      : buffer_detail(hostData, range, false) {}

  /**
   * Creates a new buffer with associated host memory.
   *
   * The host accesses can be read-only.
   * However, the typename DataType is not const,
   * so the device accesses can be both read and write accesses.
   * Since the hostData is const,
   * this buffer is only initialized with this memory
   * and there is no write after its destruction,
   * unless there is another final data address given
   * after construction of the buffer.
   * The default value of the allocator is going to be the buffer_allocator
   * which will be of type DataType.
   * @param hostData points to the storage and values used by the buffer
   * @param range<dimensions> defines the size.
   */
  buffer_detail(const DataType* hostData, range<dimensions> range)
      : buffer_detail(const_cast<DataType*>(hostData),  // NOLINT
                      range, true) {}

  /**
   * Create a new buffer of the given size with storage managed by the SYCL
  // runtime.
   * The default behavior is to use the default host buffer allocator,
   * in order to allow for host accesses.
   * If the type of the buffer has the const qualifier,
   * then the default allocator will remove the qualifier
   * to allow host access to the data.
   * @param range<dimensions> defines the size.
   */
  buffer_detail(const range<dimensions>& range)
      : host_data(ptr_t(new DataType[range.size()])),
        rang(range),
        is_read_only(false),
        is_blocking(false) {}

  /**
   * Create a new buffer with associated memory, using the data in hostData.
   * The ownership of the hostData is shared between the runtime and the user.
   * In order to enable both the user application and the SYCL runtime
   * to use the same pointer, a mutex_class is used.
   * The mutex m is locked by the runtime whenever the data is in use
   * and unlocked otherwise.
   * Data is synchronized with hostData, when the mutex is unlocked by the
   * runtime.
   */
  buffer_detail(shared_ptr_class<DataType>& hostData,
                const range<dimensions>& bufferRange, mutex_class* m);

  /**
   * Create a new buffer which is initialized by hostData.
   * The SYCL runtime receives full ownership of the hostData unique_ptr
   * and in effect there is no synchronization with the application code
   * using hostData.
   */
  buffer_detail(unique_ptr_class<void>&& hostData,
                const range<dimensions>& bufferRange);

  // TODO(progtx):
  /**
   * Create a new sub-buffer without allocation to have separate accessors
   * later.
   * @param b is the buffer with the real data.
   * @param baseIndex specifies the origin of the sub-buffer inside the buffer
   * b.
   * @param subRange specifies the size of the sub-buffer.
   */
  buffer_detail(buffer_detail& b, const id<dimensions>& baseIndex,
                const range<dimensions>& subRange)
      : rang(subRange),
        is_read_only(b.is_read_only),
        is_blocking(b.is_blocking) {
    DataType* start = b.host_data.get();

    if (dimensions == 1) {
      start += static_cast<::size_t>(baseIndex.get(0));
    } else if (dimensions == 2) {
      start += static_cast<::size_t>(baseIndex.get(1)) *
                   static_cast<::size_t>(rang.get(0)) +
               static_cast<::size_t>(baseIndex.get(0));
    } else if (dimensions == 3) {
      // TODO(progtx):
    }

    host_data = ptr_t(start, [](DataType* ptr) {});
  }

  /**
   * Creates a buffer from an existing OpenCL memory object associated to a
   * context
   * after waiting for an event signaling the availability of the OpenCL data.
   * @param mem_object is the OpenCL memory object to use.
   * @param from_queue is the queue associated to the memory object.
   * @param available_event specifies the event to wait for if non null
   */
  buffer_detail(cl_mem mem_object, queue& from_queue,
                event available_event = {});

  buffer_detail(const buffer_detail&) = default;
  buffer_detail(buffer_detail&&) noexcept = default;  // NOLINT
  buffer_detail& operator=(const buffer_detail&) = default;
  buffer_detail& operator=(buffer_detail&&) = default;  // NOLINT

  ~buffer_detail() {
    event::wait_and_throw(events);
  }

  /**
   * Return a range object representing the size of the buffer
   * in terms of number of elements in each dimension as passed to the
   * constructor.
   */
  range<dimensions> get_range() {
    return rang;
  }

  /** Total number of elements in the buffer */
  ::size_t get_count() const {
    ::size_t count = rang.get(0);
    for (int i = 1; i < dimensions; ++i) {
      count *= rang.get(i);
    }
    return count;
  }

  /** Total number of bytes in the buffer */
  ::size_t get_size() const {
    return get_count() * data_size<DataType_t>::get();
  }

 private:
  static void create(queue* q, const vector_class<cl_event>& wait_events,
                     buffer_detail* buffer) {
    ::cl_int error_code;
    const cl_mem_flags all_flags =
        ((buffer->host_data == nullptr) ? 0 : CL_MEM_USE_HOST_PTR) |
        (buffer->is_read_only ? CL_MEM_READ_ONLY : CL_MEM_READ_WRITE);
    buffer->device_data = buffer_base::cl_create_buffer(
        q, all_flags, buffer->get_size(), buffer->host_data.get(), error_code);
    detail::error::report(error_code);
    buffer->device_data.release_one();
  }

  void init() {
    if (!is_initialized) {
      command::group_detail::add_buffer_init(create, __func__, this);
      is_initialized = true;
    }
  }

  void check_read_only() {
    if (is_read_only) {
      detail::error::report(error::code::TRYING_TO_WRITE_READ_ONLY_BUFFER);
    }
  }

  template <access::mode mode, access::target target>
  using acc_return_t = accessor<DataType_t, dimensions, mode, target>;

  template <access::mode mode, access::target target>
  acc_return_t<mode, target> get_access_device(handler& cgh) {
    command::group_detail::check_scope();
    if (mode != access::mode::read) {
      check_read_only();
    }
    init();
    command::group_detail::add_buffer_access(buffer_access{this, mode, target},
                                             __func__);
    return acc_return_t<mode, target>(
        *(static_cast<cl::sycl::buffer<DataType_t, dimensions>*>(this)), cgh);
  }

  template <access::mode mode, access::target target>
  acc_return_t<mode, target> get_access_host() {
    if (mode != access::mode::read) {
      check_read_only();
    }
    return acc_return_t<mode, target>(
        *(static_cast<cl::sycl::buffer<DataType_t, dimensions>*>(this)));
  }

 public:
  template <access::mode mode,
            access::target target = access::target::global_buffer>
  accessor<DataType_t, dimensions, mode, target> get_access(handler& cgh) {
    return get_access_device<mode, target>(cgh);
  }

  template <access::mode mode, access::target target>
  accessor<DataType_t, dimensions, mode, target> get_access() {
    return get_access_host<mode, target>();
  }

 private:
  // TODO(progtx):
  void enqueue(queue* q, const vector_class<cl_event>& wait_events,
               clEnqueueBuffer_f clEnqueueBuffer) final {
    cl_event evnt;
    auto error_code = this->cl_enqueue_buffer(
        q, get_size(), host_data.get(), wait_events, evnt, clEnqueueBuffer);
    detail::error::report(error_code);
    events.emplace_back(evnt);
  }

 protected:
  template <info::detail::buffer param>
  param_traits_t<info::detail::buffer, param> get_info() const {
    return detail::non_vector_traits<info::detail::buffer, param, 1>::get(
        device_data.get());
  }

 public:
  void set_final_data(weak_ptr_class<DataType_t>& finalData);

  // TODO(progtx): nullptr indicates not to copy back
  void set_final_data(std::nullptr_t) {}
};

}  // namespace detail

#if MSVC_2013_OR_LOWER
#define BUFFER_INHERIT_CONSTRUCTORS(dimensions)                            \
  buffer(const range<dimensions>& range) : Base(range) {}                  \
  buffer(DataType* host_data, range<dimensions> range)                     \
      : Base(host_data, range) {}                                          \
  buffer(const DataType* host_data, range<dimensions> range)               \
      : Base(host_data, range) {}                                          \
  buffer(shared_ptr_class<DataType>& hostData,                             \
         const range<dimensions>& bufferRange, mutex_class* m)             \
      : Base(hostData, bufferRange, m) {}                                  \
  buffer(unique_ptr_class<void>&& hostData,                                \
         const range<dimensions>& bufferRange)                             \
      : Base(hostData, bufferRange) {}                                     \
  buffer(buffer& b, const id<dimensions>& baseIndex,                       \
         const range<dimensions>& subRange)                                \
      : Base(b, baseIndex, subRange) {}                                    \
  buffer(cl_mem mem_object, queue& from_queue, event available_event = {}) \
      : Base(mem_object, from_queue, available_event) {}
#endif

template <typename DataType_t>
struct buffer<DataType_t, 1> : public detail::buffer_detail<DataType_t, 1> {
 private:
  using Base = detail::buffer_detail<DataType_t, 1>;
  using DataType = typename Base::value_type;

 public:
#if MSVC_2013_OR_LOWER
  BUFFER_INHERIT_CONSTRUCTORS(1)
#else
  using Base::Base;
#endif

  /**
   * Create a new allocated 1D buffer initialized from the given elements
   * ranging from first up to one before last
   */
  template <class InputIterator>
  buffer(InputIterator first, InputIterator last)
      : Base(nullptr, last - first) {
    this->host_data = this->ptr_t(new DataType[last - first]);
    std::copy(first, last, this->host_data.get());
  }

  buffer(vector_class<DataType>& host_data)
      : Base(host_data.data(), host_data.size()) {}
  buffer(const vector_class<DataType>& host_data)
      : Base(host_data.data(), host_data.size()) {}
};

template <typename DataType_t>
struct buffer<DataType_t, 2> : public detail::buffer_detail<DataType_t, 2> {
 private:
  using Base = detail::buffer_detail<DataType_t, 2>;
  using DataType = typename Base::value_type;

 public:
#if MSVC_2013_OR_LOWER
  BUFFER_INHERIT_CONSTRUCTORS(2)
#else
  using Base::Base;
#endif
  buffer(::size_t sizeX, ::size_t sizeY) : buffer(range<2>{sizeX, sizeY}) {}
  buffer(DataType* host_data, ::size_t sizeX, ::size_t sizeY)
      : buffer(host_data, {sizeX, sizeY}) {}
  buffer(const DataType* host_data, ::size_t sizeX, ::size_t sizeY)
      : buffer(host_data, {sizeX, sizeY}) {}
};

template <typename DataType_t>
struct buffer<DataType_t, 3> : public detail::buffer_detail<DataType_t, 3> {
 private:
  using Base = detail::buffer_detail<DataType_t, 3>;
  using DataType = typename Base::value_type;

 public:
#if MSVC_2013_OR_LOWER
  BUFFER_INHERIT_CONSTRUCTORS(3)
#else
  using Base::Base;
#endif
  buffer(::size_t sizeX, ::size_t sizeY, ::size_t sizeZ)
      : buffer(range<3>{sizeX, sizeY, sizeZ}) {}
  buffer(DataType* host_data, ::size_t sizeX, ::size_t sizeY, ::size_t sizeZ)
      : buffer(host_data, {sizeX, sizeY, sizeZ}) {}
  buffer(const DataType* host_data, ::size_t sizeX, ::size_t sizeY,
         ::size_t sizeZ)
      : buffer(host_data, {sizeX, sizeY, sizeZ}) {}
};

#if MSVC_2013_OR_LOWER
#undef BUFFER_INHERIT_CONSTRUCTORS
#endif

}  // namespace sycl
}  // namespace cl
