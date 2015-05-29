#pragma once

// 3.6.1 Buffers

#include "access.h"
#include "error_handler.h"
#include "event.h"
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
class command_group;
class queue;

namespace detail {

// Forward declarations
template <typename DataType, int dimensions>
class accessor_buffer;
namespace kernel_ {
	class source;
}

class buffer_base {
protected:
	friend class kernel_::source;

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

struct buffer_access {
	buffer_base* data;
	access::mode mode;
	access::target target;
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
	// Associated host memory.
	// The buffer will use this host memory for its full lifetime,
	// but the contents of this host memory are undefined for the lifetime of the buffer.
	// If the host memory is modified by the host, or mapped to another buffer or image during the lifetime of this buffer,
	// then the results are undefined.
	// The initial contents of the buffer will be the contents of the host memory at the time of construction.
	// When the buffer is destroyed, the destructor will block until all work in queues on the buffer has completed,
	// then copy the contents of the buffer back to the host memory (if required) and then return.
	buffer_(DataType* host_data, range<dimensions> range)
		: buffer_(host_data, range, false) {}

	// Associated host memory, read-only mode.
	buffer_(const DataType* host_data, range<dimensions> range)
		: buffer_(host_data, range, true) {}

	// No associated storage.
	// The storage for this type of buffer is entirely handled by the SYCL system.
	// The destructor for this type of buffer never blocks, even if work on the buffer has not completed.
	// Instead, the SYCL system frees any storage required for the buffer asynchronously when it is no longer in use in queues.
	// The initial contents of the buffer are undefined.
	buffer_(range<dimensions> range)
		:	host_data(ptr_t(new DataType[ range[0]*range[1]*range[2] ])),
			rang(range),
			is_read_only(false),
			is_blocking(false) {}

	// Associated storage object.
	// The storage object must not be destroyed by the user until after the buffer has been destroyed.
	// The synchronization and copying behavior of the storage is determined by the storage object.
	//buffer(storage<DataType>& store, range<dimensions>);

	// Creates a sub-buffer object, which is a sub-range reference to a base buffer.
	// This sub-buffer can be used to create accessors to the base buffer,
	// but which only have access to the range specified at time of construction of the sub-buffer.
	//buffer_(buffer_, index<dimensions> base_index, range<dimensions> sub_range);

	// Creates a buffer from an existing OpenCL memory object associated to a context
	// after waiting for an event signaling the availability of the OpenCL data.
	// mem_object is the OpenCL memory object to use.
	// from_queue is the queue associated to the memory object.
	// available_event specifies the event to wait for if non null
	buffer_(cl_mem mem_object, queue from_queue, event available_event);

	// Return a range object representing the size of the buffer
	// in terms of number of elements in each dimension as passed to the constructor.
	range<dimensions> get_range() {
		return rang;
	}

	// Total number of elements in the buffer
	size_t get_count() const {
		size_t count = rang[0];
		for(int i = 1; i < dimensions; ++i) {
			count *= rang[i];
		}
		return count;
	}

	// Total number of bytes in the buffer
	size_t get_size() const {
		return get_count() * sizeof(DataType);
	}

private:
	template<cl_mem_flags FLAGS>
	static void create(queue* q, buffer_* buffer) {
		cl_int error_code;
		const cl_mem_flags all_flags = FLAGS | ((buffer->host_data == nullptr) ? 0 : CL_MEM_USE_HOST_PTR);
		buffer->device_data = clCreateBuffer(q->get_context().get(), all_flags, buffer->get_size(), buffer->host_data.get(), &error_code);
		detail::error::report(error_code);
	}

	template<cl_mem_flags FLAGS>
	void init() {
		if(!is_initialized) {
			command::group_::add(create<FLAGS>, __func__, this);
			is_initialized = true;
		}
	}

	void check_write() {
		if(is_read_only) {
			detail::error::report(error::code::TRYING_TO_WRITE_READ_ONLY_BUFFER);
		}
	}

	template<access::mode mode, access::target target>
	accessor<DataType, dimensions, mode, target> create_accessor() {
		if(command::group_::in_scope()) {
			command::group_::add(buffer_access{ this, mode, target }, __func__);
		}
		return accessor<DataType, dimensions, mode, target>(*(reinterpret_cast<cl::sycl::buffer<DataType, dimensions>*>(this)));
	}

public:
	// Default access is read-only
	template<access::mode mode, access::target target = access::global_buffer>
	accessor<DataType, dimensions, mode, target> get_access() {
		if(target != access::host_buffer) {
			command::group_::check_scope();
			init<CL_MEM_READ_ONLY>();
		}
		return create_accessor<mode, target>();
	}

	// TODO: Handle multiple access requests
#define SYCL_GET_ACCESS(mode, target, flags, code)				\
	template<>													\
	accessor<DataType, dimensions, mode, target> get_access() {	\
		if(target != access::host_buffer) {						\
			command::group_::check_scope();						\
		}														\
		code;													\
		if(target != access::host_buffer) {						\
			init<flags>();										\
		}														\
		return create_accessor<mode, target>();					\
	}

	// TODO: Implement other combinations
	SYCL_GET_ACCESS(access::write, access::global_buffer, CL_MEM_WRITE_ONLY, check_write());
	SYCL_GET_ACCESS(access::read_write, access::global_buffer, CL_MEM_READ_WRITE, check_write());

#undef SYCL_GET_ACCESS

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
	template<cl_mem_info name>
	using parameter_t = typename param_traits<cl_mem_info, name>::param_type;

	template<cl_mem_info name>
	parameter_t<name> get_info() const {
		parameter_t<name> param_value;
		auto mem = device_data.get();
		auto error_code = clGetMemObjectInfo(mem, name, sizeof(parameter_t<name>), &param_value, nullptr);
		detail::error::report(error_code);
		return param_value;
	}
};

} // namespace detail

// Defines a shared array that can be used by kernels in queues and has to be accessed using accessor classes.

template <typename DataType>
struct buffer<DataType, 1> : public detail::buffer_<DataType, 1> {
private:
	using Base = detail::buffer_<DataType, 1>;
public:
#if MSVC_LOW
	buffer(range<1> range)
		: Base(range) {}
	buffer(DataType* host_data, range<1> range)
		: Base(host_data, range) {}
	buffer(const DataType* host_data, range<1> range)
		: Base(host_data, range) {}
	//buffer(storage<DataType>& store, range<1>);
	//buffer(buffer, index<dimensions> base_index, range<dimensions> sub_range);
	buffer(cl_mem mem_object, queue from_queue, event available_event)
		: Base(mem_object, from_queue, available_event) {}
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
	buffer(range<2> range)
		: Base(range) {}
	buffer(DataType* host_data, range<2> range)
		: Base(host_data, range) {}
	buffer(const DataType* host_data, range<2> range)
		: Base(host_data, range) {}
	//buffer(storage<DataType>& store, range<2>);
	//buffer(buffer, index<2> base_index, range<2> sub_range);
	buffer(cl_mem mem_object, queue from_queue, event available_event)
		: Base(mem_object, from_queue, available_event) {}
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
	buffer(range<3> range)
		: Base(range) {}
	buffer(DataType* host_data, range<3> range)
		: Base(host_data, range) {}
	buffer(const DataType* host_data, range<3> range)
		: Base(host_data, range) {}
	//buffer(storage<DataType>& store, range<3>);
	//buffer(buffer, index<3> base_index, range<3> sub_range);
	buffer(cl_mem mem_object, queue from_queue, event available_event)
		: Base(mem_object, from_queue, available_event) {}
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

} // namespace sycl
} // namespace cl
