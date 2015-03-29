#pragma once

// 3.6.4.4 Buffer accessors

#include "../accessor.h"

namespace cl {
namespace sycl {

namespace detail {

// Forward declaration
template <typename DataType, int dimensions>
struct cl::sycl::buffer;

template <typename DataType, int dimensions>
class accessor_buffer {
protected:
	cl::sycl::buffer<DataType, dimensions>* buf;
public:
	accessor_buffer(
		cl::sycl::buffer<DataType, dimensions>& bufferRef,
		range<dimensions> offset,
		range<dimensions> range
	) : buf(&bufferRef) {
		DSELF() << "not implemented";
	}
protected:
	cl_mem get_buffer_object() const {
		return buf->device_data.get();
	}
};

SYCL_ACCESSOR_CLASS(
	target == access::cl_buffer ||
	target == access::constant_buffer ||
	target == access::global_buffer ||
	target == access::host_buffer
), public accessor_buffer<DataType, dimensions> {
public:
	// This accessor limits the processing of the buffer to the [offset, offset + range] for every dimension
	// Any other parts of the buffer will be unaffected
	accessor_(
		cl::sycl::buffer<DataType, dimensions>& bufferRef,
		range<dimensions> offset,
		range<dimensions> range
	) : accessor_buffer(bufferRef, offset, range) {}

	accessor_(cl::sycl::buffer<DataType, dimensions>& bufferRef)
		: accessor_(
		bufferRef,
		detail::empty_range<dimensions>(),
		bufferRef.get_range()
	) {}

	virtual cl_mem get_cl_mem_object() const override {
		return get_buffer_object();
	}

	detail::data_ref operator[](int index) const {
		detail::kernel_::source::register_resource(*this);
		return detail::data_ref(
			get_resource_name() + "[" + std::to_string(index) + "]"
		);
	}

	detail::data_ref operator[](id<dimensions> index) const {
		detail::kernel_::source::register_resource(*this);
		return detail::data_ref(
			get_resource_name() + "[" + detail::kernel_::source::get_name(index) + "]"
		);
	}

protected:
	virtual string_class get_resource_name() const override {
		return obtain_resource_name(buf);
	}

	virtual void* resource() const override {
		return buf;
	}
};

} // namespace detail


SYCL_ADD_ACCESSOR(access::read) {
public:
#if MSVC_LOW
	accessor(buffer<DataType, dimensions>& targette)
		: detail::accessor_<DataType, dimensions, access::read, target>(targette) {}
#else
	using detail::accessor_<DataType, dimensions, access::read, target>::accessor_;
#endif
};

SYCL_ADD_ACCESSOR(access::write) {
public:
	accessor(buffer<DataType, dimensions>& targette)
		: detail::accessor_<DataType, dimensions, access::write, target>(targette) {}
};

SYCL_ADD_ACCESSOR(access::atomic) {
public:
	accessor(buffer<DataType, dimensions>& targette)
		: detail::accessor_<DataType, dimensions, access::atomic, target>(targette) {}
};

} // namespace sycl
} // namespace cl

#undef SYCL_ACCESSOR_CLASS
#undef SYCL_ADD_ACCESSOR
