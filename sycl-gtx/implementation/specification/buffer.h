#pragma once

// 3.4.2 Buffers

#include "access.h"
#include "error_handler.h"
#include "event.h"
#include "info.h"
#include "param_traits.h"
#include "ranges.h"
#include "refc.h"
#include "../common.h"
#include "../debug.h"
#include <algorithm>
#include <vector>


namespace cl {
namespace sycl {

// Forward declarations
template <typename DataType, int dimensions, access::mode mode, access::target target>
class accessor;
template <typename DataType, int dimensions = 1>
struct buffer;
class handler;
class queue;

namespace detail {

// Forward declarations
#if MSVC_LOW
template <typename DataType, int dimensions, int mode, int target, typename>
class accessor_;
#endif
template <typename DataType, int dimensions>
class accessor_buffer;
class command_group;
class issue_command;

#undef SYCL_ADD_ACCESS_MODE_HELPER


class buffer_base {
protected:
	friend class issue_command;

	detail::refc<cl_mem, clRetainMemObject, clReleaseMemObject> device_data;

	void create_accessor_command();

	using clEnqueueBuffer_f = decltype(&clEnqueueWriteBuffer);
	virtual void enqueue(queue* q, clEnqueueBuffer_f clEnqueueBuffer) {
		DSELF() << "not implemented";
	}
	static void enqueue_command(queue* q, buffer_base* buffer, clEnqueueBuffer_f clEnqueueBuffer) {
		buffer->enqueue(q, clEnqueueBuffer);
	}
};

template <typename DataType, int dimensions>
class buffer_ : public buffer_base {
protected:
	using ptr_t = shared_ptr_class<DataType>;

	range<dimensions> rang;
	ptr_t host_data;

	bool is_read_only = false;
	bool is_blocking = true;
	bool is_initialized = false;

	friend class accessor_base;
	friend class accessor_buffer<DataType, dimensions>;
	friend class kernel_::source;

	// Associated host memory.
	buffer_(DataType* host_data, range<dimensions> range, bool is_read_only, bool is_blocking = true)
		:	host_data(ptr_t(host_data, [](DataType* ptr) {})),
			rang(range),
			is_read_only(is_read_only),
			is_blocking(is_blocking) {}

	buffer_(nullptr_t host_data, range<dimensions> range)
		: buffer_(nullptr, range, false) {}

public:
	using value_type = DataType;
	using reference = value_type&;
	using const_reference = const value_type&;

	// Creates a new buffer with associated host memory.
	// The memory is owned by the runtime during the lifetime of the object.
	// Data is copied back to the host unless the user overrides the behavior using the set_final_data method.
	// hostData points to the storage and values used by the buffer and range<dimensions> defines the size.
	buffer_(DataType* hostData, range<dimensions> range)
		: buffer_(hostData, range, false) {}

	// Creates a new buffer with associated host memory.
	// hostData points to the storage and values used by the buffer and range<dimensions> defines the size.
	// The host accesses can be read-only.
	// However, the typename DataType is not const so the device accesses can be both read and write accesses.
	// Since the hostData is const, this buffer is only initialized with this memory and there is no write after its destruction,
	// unless there is another final data address given after construction of the buffer.
	// The default value of the allocator is going to be the buffer_allocator which will be of type DataType.
	buffer_(const DataType* hostData, range<dimensions> range)
		: buffer_(hostData, range, true) {}

	// Create a new buffer of the given size with storage managed by the SYCL runtime.
	// The default behavior is to use the default host buffer allocator,
	// in order to allow for host accesses.
	// If the type of the buffer has the const qualifier,
	// then the default allocator will remove the qualifier to allow host access to the data.
	buffer_(const range<dimensions>& range)
		:	host_data(ptr_t(new DataType[ detail::get_size(range) ])),
			rang(range),
			is_read_only(false),
			is_blocking(false) {}

	// Create a new buffer with associated memory, using the data in hostData.
	// The ownership of the hostData is shared between the runtime and the user.
	// In order to enable both the user application and the SYCL runtime to use the same pointer, a mutex_class is used.
	// The mutex m is locked by the runtime whenever the data is in use and unlocked otherwise.
	// Data is synchronized with hostData, when the mutex is unlocked by the runtime.
	buffer_(shared_ptr_class<DataType>& hostData, const range<dimensions>& bufferRange, mutex_class * m);

	// Create a new buffer which is initialized by hostData.
	// The SYCL runtime receives full ownership of the hostData unique_ptr
	// and in effect there is no synchronization with the application code using hostData.
	buffer_(unique_ptr_class<void>&& hostData, const range<dimensions>& bufferRange);
	
	// Create a new sub-buffer without allocation to have separate accessors later.
	// b is the buffer with the real data.
	// baseIndex specifies the origin of the sub-buffer inside the buffer b.
	// subRange specifies the size of the sub-buffer.
	buffer_(buffer_& b, const index<dimensions>& baseIndex, const range<dimensions>& subRange);

	// Creates a buffer from an existing OpenCL memory object associated to a context
	// after waiting for an event signaling the availability of the OpenCL data.
	// mem_object is the OpenCL memory object to use.
	// from_queue is the queue associated to the memory object.
	// available_event specifies the event to wait for if non null
	buffer_(cl_mem mem_object, queue& from_queue, event available_event = {});

	// Return a range object representing the size of the buffer
	// in terms of number of elements in each dimension as passed to the constructor.
	range<dimensions> get_range() {
		return rang;
	}

	// Total number of elements in the buffer
	size_t get_count() const {
		size_t count = rang.get(0);
		for(int i = 1; i < dimensions; ++i) {
			count *= rang.get(i);
		}
		return count;
	}

	// Total number of bytes in the buffer
	size_t get_size() const {
		return get_count() * sizeof(DataType);
	}

private:
	static void create(queue* q, buffer_* buffer) {
		cl_int error_code;
		const cl_mem_flags all_flags =
			((buffer->host_data == nullptr) ? 0 : CL_MEM_USE_HOST_PTR)	|
			(buffer->is_read_only ? CL_MEM_READ_ONLY : CL_MEM_READ_WRITE);
		buffer->device_data = clCreateBuffer(q->get_context().get(), all_flags, buffer->get_size(), buffer->host_data.get(), &error_code);
		detail::error::report(error_code);
	}

	void init() {
		if(!is_initialized) {
			command::group_::add(create, __func__, this);
			is_initialized = true;
		}
	}

	void check_read_only() {
		if(is_read_only) {
			detail::error::report(error::code::TRYING_TO_WRITE_READ_ONLY_BUFFER);
		}
	}

#if MSVC_LOW
	// Indirection required because MSVC2013 fails on enum parameter SFINAE
	// The final accessor class also needs a move constructor from base type: accessor(accessor_&&)
	template <int mode, int target>
	using acc_return_t = accessor_<DataType, dimensions, mode, target>;
#else
	template <access::mode mode, access::target target>
	using acc_return_t = accessor<DataType, dimensions, mode, target>;
#endif

	template <int mode, int target, class = typename std::enable_if<target == access::global_buffer>::type>
	acc_return_t<mode, target> get_access_device(handler& cgh) {
		command::group_::check_scope();
		if(mode != access::read) {
			check_read_only();
		}
		init();
		command::group_::add(buffer_access{ this, (access::mode)mode, (access::target)target }, __func__);
		return acc_return_t<mode, target>(*(reinterpret_cast<cl::sycl::buffer<DataType, dimensions>*>(this)), &cgh);
	}

	template <int mode, int target, class = typename std::enable_if<target == access::host_buffer>::type>
	acc_return_t<mode, target> get_access_host() {
		if(mode != access::read) {
			check_read_only();
		}
		return acc_return_t<mode, target>(*(reinterpret_cast<cl::sycl::buffer<DataType, dimensions>*>(this)));
	}

public:
	template <access::mode mode, access::target target = access::global_buffer>
	accessor<DataType, dimensions, mode, target> get_access(handler& cgh) {
		return get_access_device<mode, target>(cgh);
	}

	template <access::mode mode, access::target target>
	accessor<DataType, dimensions, mode, target> get_access() {
		return get_access_host<mode, target>();
	}

private:
	// TODO
	virtual void enqueue(queue* q, clEnqueueBuffer_f clEnqueueBuffer) override {
		cl_int error_code = clEnqueueBuffer(
			q->get(),
			device_data.get(),
			// TODO: It shouldn't block here, SYCL runtime needs to take care of consistency
			true,
			// TODO: Sub-buffer access
			0, get_size(),
			// TODO: Events
			host_data.get(), 0, nullptr, nullptr
		);
		detail::error::report(error_code);
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
	void set_final_data(weak_ptr_class<DataType>& finalData);
};

} // namespace detail

#if MSVC_LOW
#define BUFFER_INHERIT_CONSTRUCTORS(dimensions)															\
	buffer(const range<dimensions>& range)																\
		: Base(range) {}																				\
	buffer(DataType* host_data, range<dimensions> range)												\
		: Base(host_data, range) {}																		\
	buffer(const DataType* host_data, range<dimensions> range)											\
		: Base(host_data, range) {}																		\
	buffer(shared_ptr_class<DataType>& hostData, const range<dimensions>& bufferRange, mutex_class* m)	\
		: Base(hostData, bufferRange, m) {}																\
	buffer(unique_ptr_class<void>&& hostData, const range<dimensions>& bufferRange)						\
		: Base(hostData, bufferRange) {}																\
	buffer(buffer& b, const index<dimensions>& baseIndex, const range<dimensions>& subRange)			\
		: Base(b, baseIndex, subRange) {}																\
	buffer(cl_mem mem_object, queue& from_queue, event available_event = {})							\
		: Base(mem_object, from_queue, available_event) {}
#endif

template <typename DataType>
struct buffer<DataType, 1> : public detail::buffer_<DataType, 1> {
private:
	using Base = detail::buffer_<DataType, 1>;
public:
#if MSVC_LOW
	BUFFER_INHERIT_CONSTRUCTORS(1)
#else
	using Base::buffer;
#endif
	// Create a new allocated 1D buffer initialized from the given elements
	// ranging from first up to one before last
	template <class InputIterator>
	buffer(InputIterator first, InputIterator last)
		: Base(nullptr, last - first) {
		host_data = ptr_t(new DataType[last - first]);
		std::copy(first, last, host_data.get());
	}

	buffer(vector_class<DataType>& host_data)
		: Base(host_data.data(), host_data.size()) {}
	buffer(const vector_class<DataType>& host_data)
		: Base(host_data.data(), host_data.size()) {}
};

template <typename DataType>
struct buffer<DataType, 2> : public detail::buffer_<DataType, 2>{
#if MSVC_LOW
private:
	using Base = detail::buffer_<DataType, 2>;
public:
	BUFFER_INHERIT_CONSTRUCTORS(2)
#else
	using detail::buffer_<DataType, 2>::buffer;
#endif
	buffer(size_t sizeX, size_t sizeY)
		: buffer(range<2>{ sizeX, sizeY }) {}
	buffer(DataType* host_data, size_t sizeX, size_t sizeY)
		: buffer(host_data, { sizeX, sizeY }) {}
	buffer(const DataType* host_data, size_t sizeX, size_t sizeY)
		: buffer(host_data, { sizeX, sizeY }) {}
};

template <typename DataType>
struct buffer<DataType, 3> : public detail::buffer_<DataType, 3>{
#if MSVC_LOW
private:
	using Base = detail::buffer_<DataType, 3>;
public:
	BUFFER_INHERIT_CONSTRUCTORS(3)
#else
	using detail::buffer_<DataType, 3>::buffer;
#endif
	buffer(size_t sizeX, size_t sizeY, size_t sizeZ)
		: buffer(range<3>{ sizeX, sizeY, sizeZ }) {}
	buffer(DataType* host_data, size_t sizeX, size_t sizeY, size_t sizeZ)
		: buffer(host_data, { sizeX, sizeY, sizeZ }) {}
	buffer(const DataType* host_data, size_t sizeX, size_t sizeY, size_t sizeZ)
		: buffer(host_data, { sizeX, sizeY, sizeZ }) {}
};

#undef BUFFER_INHERIT_CONSTRUCTORS

} // namespace sycl
} // namespace cl
