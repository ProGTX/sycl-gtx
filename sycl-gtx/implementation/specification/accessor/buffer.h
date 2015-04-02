#pragma once

// 3.6.4.4 Buffer accessors

#include "../accessor.h"

namespace cl {
namespace sycl {

// Forward declaration
template <typename DataType, int dimensions>
struct buffer;

namespace detail {

template <typename DataType, int dimensions>
class accessor_buffer {
protected:
	cl::sycl::buffer<DataType, dimensions>* buf;
	range<dimensions> offset;
	range<dimensions> rang;
public:
	accessor_buffer(
		cl::sycl::buffer<DataType, dimensions>& bufferRef,
		range<dimensions> offset,
		range<dimensions> range
	) : buf(&bufferRef), offset(offset), rang(range) {
		DSELF() << "not implemented";
	}
protected:
	cl_mem get_buffer_object() const {
		return buf->device_data.get();
	}
};

#define SYCL_BUFFER_CONSTRUCTORS(init)					\
	accessor_(														\
		cl::sycl::buffer<DataType, dimensions>& bufferRef,			\
		range<dimensions> offset,									\
		range<dimensions> range										\
	) : accessor_buffer(bufferRef, offset, range) {					\
		init														\
	}																\
	accessor_(cl::sycl::buffer<DataType, dimensions>& bufferRef)	\
		: accessor_(												\
		bufferRef,													\
		detail::empty_range<dimensions>(),							\
		bufferRef.get_range()										\
	) {}

SYCL_ACCESSOR_CLASS(
	target == access::cl_buffer ||
	target == access::constant_buffer ||
	target == access::global_buffer
), public accessor_buffer<DataType, dimensions> {
public:
	SYCL_BUFFER_CONSTRUCTORS({});

	virtual cl_mem get_cl_mem_object() const override {
		return get_buffer_object();
	}

	detail::data_ref operator[](int index) const {
		detail::kernel_::source::register_resource(*this);
		return detail::data_ref(
			get_resource_name() + "[" + std::to_string(index) + "]"
		);
	}

	// TODO: Limit id to same dimension as buffer
	detail::data_ref operator[](id<1> index) const {
		detail::kernel_::source::register_resource(*this);
		return detail::data_ref(
			get_resource_name() + "[" + index[0].name + "]"
		);
	}
	detail::data_ref operator[](id<2> index) const {
		detail::kernel_::source::register_resource(*this);
		return detail::data_ref(
			get_resource_name() + "[" + id_base_all_name + "]"
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

template <int level, typename DataType, int dimensions, access::mode mode>
class accessor_host_ref {
protected:
	using Lower = accessor_host_ref<dimensions - 1, DataType, dimensions, mode>;
	using acc_t = accessor_<DataType, dimensions, mode, access::host_buffer>;
	friend class acc_t;
	friend class accessor_host_ref;
	acc_t* acc;
	accessor_host_ref(acc_t* acc = nullptr)
		: acc(acc) {}
public:
	Lower operator[](int index) {
		return Lower(acc);
	}
};

SYCL_ACCESSOR_CLASS(target == access::host_buffer),
	public accessor_buffer<DataType, dimensions>,
	public accessor_host_ref<dimensions, DataType, dimensions, (access::mode)mode>
{
public:
	SYCL_BUFFER_CONSTRUCTORS({
		acc = this;
	});
};

} // namespace detail


SYCL_ADD_ACCESSOR(access::read) {
	using Base = detail::accessor_<DataType, dimensions, access::read, target>;
public:
#if MSVC_LOW
	accessor(buffer<DataType, dimensions>& bufferRef)
		: Base(bufferRef) {}
	accessor(
		buffer<DataType, dimensions>& bufferRef,
		range<dimensions> offset,
		range<dimensions> range
	)
		: Base(bufferRef, offset, range) {}
#else
	using Base::accessor_;
#endif
};

SYCL_ADD_ACCESSOR(access::write) {
	using Base = detail::accessor_<DataType, dimensions, access::write, target>;
public:
#if MSVC_LOW
	accessor(buffer<DataType, dimensions>& bufferRef)
		: Base(bufferRef) {}
	accessor(
		buffer<DataType, dimensions>& bufferRef,
		range<dimensions> offset,
		range<dimensions> range
	)
		: Base(bufferRef, offset, range) {}
#else
	using Base::accessor_;
#endif
};

} // namespace sycl
} // namespace cl

#undef SYCL_BUFFER_CONSTRUCTORS

#undef SYCL_ACCESSOR_CLASS
#undef SYCL_ADD_ACCESSOR
