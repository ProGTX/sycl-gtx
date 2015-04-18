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
	size_t access_buffer_range(int n) const {
		return buf->rang[n];
	}
	DataType* access_host_data() const {
		return buf->host_data.get();
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

template <int dimensions>
struct id_name {
	static string_class get(id<dimensions> index) {
		return id_base_all_name;
	}
};
template <>
struct id_name<1> {
	static string_class get(id<1> index) {
		return index[0].name;
	}
};

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

	detail::data_ref operator[](id<dimensions> index) const {
		detail::kernel_::source::register_resource(*this);
		return detail::data_ref(
			get_resource_name() + "[" + id_name<dimensions>::get(index) + "]"
		);
	}

	detail::data_ref operator[](detail::data_ref ref) const {
		detail::kernel_::source::register_resource(*this);
		return detail::data_ref(
			get_resource_name() + "[" + ref.name + "]"
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

#define SYCL_ACCESSOR_HOST_REF_CONSTRUCTOR()									\
	using acc_t = accessor_<DataType, dimensions, mode, access::host_buffer>;	\
	friend class acc_t;															\
	friend class accessor_host_ref;												\
	acc_t* acc;																	\
	range<3> rang;																\
	accessor_host_ref(acc_t* acc, range<3> range)								\
		: acc(acc), rang(range) {}												\
	accessor_host_ref()															\
		: accessor_host_ref(nullptr, empty_range<3>()) {}

template <int level, typename DataType, int dimensions, access::mode mode>
class accessor_host_ref {
protected:
	using Lower = accessor_host_ref<dimensions - 1, DataType, dimensions, mode>;
	SYCL_ACCESSOR_HOST_REF_CONSTRUCTOR();
public:
	Lower operator[](int index) {
		auto rang_copy = rang;
		rang_copy[dimensions - level] = index;
		return Lower(acc, rang_copy);
	}
};

template <typename DataType, int dimensions, access::mode mode>
class accessor_host_ref<1, DataType, dimensions, mode> {
protected:
	SYCL_ACCESSOR_HOST_REF_CONSTRUCTOR();
public:
	DataType& operator[](int index) {
		// http://stackoverflow.com/questions/7367770
		rang[dimensions - 1] = index;
		index = 0;
		int multiplier = 1;
		for(int i = 0; i < dimensions; ++i) {
			index += rang[i] * multiplier;
			multiplier *= acc->access_buffer_range(i);
		}
		return acc->access_host_data()[index];
	}
};

SYCL_ACCESSOR_CLASS(target == access::host_buffer),
	public accessor_buffer<DataType, dimensions>,
	public accessor_host_ref<dimensions, DataType, dimensions, (access::mode)mode>
{
	template <int level, typename DataType, int dimensions, access::mode mode>
	friend class accessor_host_ref;
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

SYCL_ADD_ACCESSOR(access::read_write) {
	using Base = detail::accessor_<DataType, dimensions, access::read_write, target>;
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

#undef SYCL_ACCESSOR_HOST_REF_CONSTRUCTOR
#undef SYCL_BUFFER_CONSTRUCTORS

#undef SYCL_ACCESSOR_CLASS
#undef SYCL_ADD_ACCESSOR
