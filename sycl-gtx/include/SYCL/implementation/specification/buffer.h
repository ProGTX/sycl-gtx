#pragma once

// 3.4.2 Buffers

#include "access.h"
#include "buffer_base.h"
#include "command_group.h"
#include "error_handler.h"
#include "event.h"
#include "info.h"
#include "param_traits.h"
#include "ranges.h"
#include "refc.h"
#include "../common.h"
#include "../debug.h"
#include "../synchronizer.h"
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
class accessor_;
template <typename, int>
class accessor_buffer;
class command_group;

#undef SYCL_ADD_ACCESS_MODE_HELPER


template <typename DataType_, int dimensions>
class buffer_ : public buffer_base {
public:
  using value_type = typename base_host_data<DataType_>::type;
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
  friend class accessor_buffer<DataType_, dimensions>;
  friend class kernel_::source;

  // Associated host memory.
  buffer_(
    value_type* host_data,
    range<dimensions> range,
    bool is_read_only,
    bool is_blocking = true)
    : host_data(ptr_t(host_data, [](value_type* ptr) {})),
      rang(range),
      is_read_only(is_read_only),
      is_blocking(is_blocking) {}

  buffer_(std::nullptr_t host_data, range<dimensions> range)
    : buffer_(nullptr, range, false) {}

public:
  // Creates a new buffer with associated host memory.
  // The memory is owned by the runtime during the lifetime of the object.
  // Data is copied back to the host unless the user overrides the behavior
  // using the set_final_data method.
  // hostData points to the storage and values used by the buffer
  // and range<dimensions> defines the size.
  buffer_(DataType* hostData, range<dimensions> range)
    : buffer_(hostData, range, false) {}

  // Creates a new buffer with associated host memory.
  // hostData points to the storage and values used by the buffer
  // and range<dimensions> defines the size.
  // The host accesses can be read-only.
  // However, the typename DataType is not const,
  // so the device accesses can be both read and write accesses.
  // Since the hostData is const,
  // this buffer is only initialized with this memory
  // and there is no write after its destruction,
  // unless there is another final data address given
  // after construction of the buffer.
  // The default value of the allocator is going to be the buffer_allocator
  // which will be of type DataType.
  buffer_(const DataType* hostData, range<dimensions> range)
    : buffer_(const_cast<DataType*>(hostData), range, true) {}

  // Create a new buffer of the given size with storage managed by the SYCL runtime.
  // The default behavior is to use the default host buffer allocator,
  // in order to allow for host accesses.
  // If the type of the buffer has the const qualifier,
  // then the default allocator will remove the qualifier
  // to allow host access to the data.
  buffer_(const range<dimensions>& range)
    : host_data(ptr_t(new DataType[range.size()])),
      rang(range),
      is_read_only(false),
      is_blocking(false) {}

  // Create a new buffer with associated memory, using the data in hostData.
  // The ownership of the hostData is shared between the runtime and the user.
  // In order to enable both the user application and the SYCL runtime
  // to use the same pointer, a mutex_class is used.
  // The mutex m is locked by the runtime whenever the data is in use
  // and unlocked otherwise.
  // Data is synchronized with hostData, when the mutex is unlocked by the runtime.
  buffer_(
    shared_ptr_class<DataType>& hostData,
    const range<dimensions>& bufferRange, mutex_class * m
  );

  // Create a new buffer which is initialized by hostData.
  // The SYCL runtime receives full ownership of the hostData unique_ptr
  // and in effect there is no synchronization with the application code
  // using hostData.
  buffer_(unique_ptr_class<void>&& hostData, const range<dimensions>& bufferRange);

  // TODO
  // Create a new sub-buffer without allocation to have separate accessors later.
  // b is the buffer with the real data.
  // baseIndex specifies the origin of the sub-buffer inside the buffer b.
  // subRange specifies the size of the sub-buffer.
  buffer_(
    buffer_& b, const id<dimensions>& baseIndex, const range<dimensions>& subRange)
    : rang(subRange), is_read_only(b.is_read_only), is_blocking(b.is_blocking) {
    DataType* start = b.host_data.get();

    if(dimensions == 1) {
      start += (::size_t)(baseIndex.get(0));
    }
    else if(dimensions == 2) {
      start +=
        (::size_t)(baseIndex.get(1)) * (::size_t)(rang.get(0)) +
        (::size_t)(baseIndex.get(0));
    }
    else if(dimensions == 3) {
      // TODO
    }

    host_data = ptr_t(start, [](DataType* ptr) {});
  }

  // Creates a buffer from an existing OpenCL memory object associated to a context
  // after waiting for an event signaling the availability of the OpenCL data.
  // mem_object is the OpenCL memory object to use.
  // from_queue is the queue associated to the memory object.
  // available_event specifies the event to wait for if non null
  buffer_(cl_mem mem_object, queue& from_queue, event available_event = {});

  ~buffer_() {
    event::wait_and_throw(events);
  }

  // Return a range object representing the size of the buffer
  // in terms of number of elements in each dimension as passed to the constructor.
  range<dimensions> get_range() {
    return rang;
  }

  // Total number of elements in the buffer
  ::size_t get_count() const {
    ::size_t count = rang.get(0);
    for(int i = 1; i < dimensions; ++i) {
      count *= rang.get(i);
    }
    return count;
  }

  // Total number of bytes in the buffer
  ::size_t get_size() const {
    return get_count() * data_size<DataType_>::get();
  }

private:
  static void create(
    queue* q, const vector_class<cl_event>& wait_events, buffer_* buffer) {
    ::cl_int error_code;
    const cl_mem_flags all_flags =
      ((buffer->host_data == nullptr) ? 0 : CL_MEM_USE_HOST_PTR) |
      (buffer->is_read_only ? CL_MEM_READ_ONLY : CL_MEM_READ_WRITE);
    buffer->device_data = buffer_base::cl_create_buffer(
      q, all_flags, buffer->get_size(), buffer->host_data.get(), error_code
    );
    detail::error::report(error_code);
    buffer->device_data.release_one();
  }

  void init() {
    if(!is_initialized) {
      command::group_::add_buffer_init(create, __func__, this);
      is_initialized = true;
    }
  }

  void check_read_only() {
    if(is_read_only) {
      detail::error::report(error::code::TRYING_TO_WRITE_READ_ONLY_BUFFER);
    }
  }

  template <access::mode mode, access::target target>
  using acc_return_t = accessor<DataType_, dimensions, mode, target>;

  template <access::mode mode, access::target target>
  acc_return_t<mode, target> get_access_device(handler& cgh) {
    command::group_::check_scope();
    if(mode != access::mode::read) {
      check_read_only();
    }
    init();
    command::group_::add_buffer_access(buffer_access{ this, mode, target }, __func__);
    return acc_return_t<mode, target>(
      *(reinterpret_cast<cl::sycl::buffer<DataType_, dimensions>*>(this)), cgh
      );
  }

  template <access::mode mode, access::target target>
  acc_return_t<mode, target> get_access_host() {
    if(mode != access::mode::read) {
      check_read_only();
    }
    return acc_return_t<mode, target>(
      *(reinterpret_cast<cl::sycl::buffer<DataType_, dimensions>*>(this))
      );
  }

public:
  template <access::mode mode, access::target target = access::target::global_buffer>
  accessor<DataType_, dimensions, mode, target> get_access(handler& cgh) {
    return get_access_device<mode, target>(cgh);
  }

  template <access::mode mode, access::target target>
  accessor<DataType_, dimensions, mode, target> get_access() {
    return get_access_host<mode, target>();
  }

private:
  // TODO
  virtual void enqueue(
    queue* q,
    const vector_class<cl_event>& wait_events,
    clEnqueueBuffer_f clEnqueueBuffer
  ) override {
    cl_event evnt;
    auto error_code = this->cl_enqueue_buffer(
      q,
      get_size(),
      host_data.get(),
      wait_events,
      evnt,
      clEnqueueBuffer
    );
    detail::error::report(error_code);
    events.emplace_back(evnt);
  }

protected:
  template <info::detail::buffer param>
  param_traits_t<info::detail::buffer, param>
    get_info() const {
    return detail::non_vector_traits<
      info::detail::buffer,
      param,
      1
    >::get(device_data.get());
  }

public:
  void set_final_data(weak_ptr_class<DataType_>& finalData);
};

} // namespace detail

#if MSVC_LOW
#define BUFFER_INHERIT_CONSTRUCTORS(dimensions)                                     \
  buffer(const range<dimensions>& range)                                            \
    : Base(range) {}                                                                \
  buffer(DataType* host_data, range<dimensions> range)                              \
    : Base(host_data, range) {}                                                     \
  buffer(const DataType* host_data, range<dimensions> range)                        \
    : Base(host_data, range) {}                                                     \
  buffer(                                                                           \
    shared_ptr_class<DataType>& hostData,                                           \
    const range<dimensions>& bufferRange, mutex_class* m)                           \
    : Base(hostData, bufferRange, m) {}                                             \
  buffer(unique_ptr_class<void>&& hostData, const range<dimensions>& bufferRange)   \
    : Base(hostData, bufferRange) {}                                                \
  buffer(                                                                           \
    buffer& b, const id<dimensions>& baseIndex, const range<dimensions>& subRange)  \
    : Base(b, baseIndex, subRange) {}                                               \
  buffer(cl_mem mem_object, queue& from_queue, event available_event = {})          \
    : Base(mem_object, from_queue, available_event) {}
#endif

template <typename DataType_>
struct buffer<DataType_, 1> : public detail::buffer_<DataType_, 1> {
private:
  using Base = detail::buffer_<DataType_, 1>;
  using DataType = typename Base::value_type;

public:
#if MSVC_LOW
  BUFFER_INHERIT_CONSTRUCTORS(1)
#else
  using Base::Base;
#endif

  // Create a new allocated 1D buffer initialized from the given elements
  // ranging from first up to one before last
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

template <typename DataType_>
struct buffer<DataType_, 2> : public detail::buffer_<DataType_, 2> {
private:
  using Base = detail::buffer_<DataType_, 2>;
  using DataType = typename Base::value_type;

public:
#if MSVC_LOW
  BUFFER_INHERIT_CONSTRUCTORS(2)
#else
  using Base::Base;
#endif
  buffer(::size_t sizeX, ::size_t sizeY)
    : buffer(range<2>{ sizeX, sizeY }) {}
  buffer(DataType* host_data, ::size_t sizeX, ::size_t sizeY)
    : buffer(host_data, { sizeX, sizeY }) {}
  buffer(const DataType* host_data, ::size_t sizeX, ::size_t sizeY)
    : buffer(host_data, { sizeX, sizeY }) {}
};

template <typename DataType_>
struct buffer<DataType_, 3> : public detail::buffer_<DataType_, 3> {
private:
  using Base = detail::buffer_<DataType_, 3>;
  using DataType = typename Base::value_type;

public:
#if MSVC_LOW
  BUFFER_INHERIT_CONSTRUCTORS(3)
#else
  using Base::Base;
#endif
  buffer(::size_t sizeX, ::size_t sizeY, ::size_t sizeZ)
    : buffer(range<3>{ sizeX, sizeY, sizeZ }) {}
  buffer(DataType* host_data, ::size_t sizeX, ::size_t sizeY, ::size_t sizeZ)
    : buffer(host_data, { sizeX, sizeY, sizeZ }) {}
  buffer(const DataType* host_data, ::size_t sizeX, ::size_t sizeY, ::size_t sizeZ)
    : buffer(host_data, { sizeX, sizeY, sizeZ }) {}
};

#if MSVC_LOW
#undef BUFFER_INHERIT_CONSTRUCTORS
#endif

} // namespace sycl
} // namespace cl
