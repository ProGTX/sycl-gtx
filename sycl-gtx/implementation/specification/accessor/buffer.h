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

#define SYCL_BUFFER_CONSTRUCTORS()									\
	accessor_(														\
		cl::sycl::buffer<DataType, dimensions>& bufferRef,			\
		range<dimensions> offset,									\
		range<dimensions> range										\
	) : accessor_buffer(bufferRef, offset, range) {					\
		acc = this;													\
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

// Forward declaration
template <int level, typename DataType, int dimensions, access::mode mode, access::target target>
class accessor_device_ref;

template <int level, typename DataType, int dimensions, access::mode mode, access::target target>
struct subscript_helper {
	using type = accessor_device_ref<level - 1, DataType, dimensions, mode, target>;
};
template <typename DataType, int dimensions, access::mode mode, access::target target>
struct subscript_helper<1, DataType, dimensions, mode, target> {
	using type = data_ref;
};

#define SYCL_ACCESSOR_DEVICE_REF_CONSTRUCTOR()							\
	using acc_t = accessor_<DataType, dimensions, mode, target>;		\
	friend class acc_t;													\
	friend class accessor_device_ref;									\
	acc_t* acc;															\
	vector_class<string_class> rang;									\
	accessor_device_ref(acc_t* acc, vector_class<string_class> range)	\
		: acc(acc), rang(range) {}										\
	accessor_device_ref()												\
		: accessor_device_ref(nullptr, {}) {							\
		rang.resize(3);													\
	}

#define SYCL_DEVICE_REF_SUBSCRIPT_OP(type)				\
	subscript_return_t operator[](type index) const {	\
		return subscript(index);						\
	}

#define SYCL_DEVICE_REF_SUBSCRIPT_OPERATORS()	\
	SYCL_DEVICE_REF_SUBSCRIPT_OP(data_ref);		\
	SYCL_DEVICE_REF_SUBSCRIPT_OP(size_t);

template <int level, typename DataType, int dimensions, access::mode mode, access::target target>
class accessor_device_ref {
protected:
	using subscript_return_t = typename subscript_helper<dimensions, DataType, dimensions, (access::mode)mode, (access::target)target>::type;
	SYCL_ACCESSOR_DEVICE_REF_CONSTRUCTOR();
	template <class T, data_ref::is_compatible_t<T>* = nullptr>
	subscript_return_t subscript(T index) const {
		auto rang_copy = rang;
		rang_copy[dimensions - level] = data_ref::get_name(index);
		return subscript_return_t(acc, rang_copy);
	}
public:
	SYCL_DEVICE_REF_SUBSCRIPT_OPERATORS();
};

template <typename DataType, int dimensions, access::mode mode, access::target target>
class accessor_device_ref<1, DataType, dimensions, mode, target> {
protected:
	using subscript_return_t = data_ref;
	SYCL_ACCESSOR_DEVICE_REF_CONSTRUCTOR();
	template <class T, data_ref::is_compatible_t<T>* = nullptr>
	data_ref subscript(T index) const {
		kernel_::source::register_resource(*acc);
		// Basically the same as with host buffer accessor, just dealing with strings
		auto rang_copy = rang;
		rang_copy[dimensions - 1] = data_ref::get_name(index);
		string_class ind(std::move(rang_copy[0]));
		auto multiplier = acc->access_buffer_range(0);
		for(int i = 1; i < dimensions; ++i) {
			ind += string_class(" + ") + std::move(rang_copy[i]) + " * " + std::to_string(multiplier);
			multiplier *= acc->access_buffer_range(i);
		}
		return data_ref(
			acc->get_resource_name() + "[" + ind + "]"
		);
	}
public:
	SYCL_DEVICE_REF_SUBSCRIPT_OPERATORS();
};

SYCL_ACCESSOR_CLASS(
	target == access::cl_buffer ||
	target == access::constant_buffer ||
	target == access::global_buffer
),
	public accessor_buffer<DataType, dimensions>,
	public accessor_device_ref<dimensions, DataType, dimensions, (access::mode)mode, (access::target)target>
{
	template <int level, typename DataType, int dimensions, access::mode mode, access::target target>
	friend class accessor_device_ref;

	using subscript_return_t = typename subscript_helper<dimensions, DataType, dimensions, (access::mode)mode, (access::target)target>::type;

public:
	SYCL_BUFFER_CONSTRUCTORS();

	virtual cl_mem get_cl_mem_object() const override {
		return get_buffer_object();
	}

	data_ref operator[](id<dimensions> index) const {
		kernel_::source::register_resource(*this);
		return data_ref(
			get_resource_name() + "[" + id_name<dimensions>::get(index) + "]"
		);
	}

	SYCL_DEVICE_REF_SUBSCRIPT_OPERATORS();

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
	SYCL_BUFFER_CONSTRUCTORS();
};

} // namespace detail

#if MSVC_LOW
#define SYCL_ADD_ACCESSOR_BUFFER(mode)										\
	SYCL_ADD_ACCESSOR(mode) {												\
		using Base = detail::accessor_<DataType, dimensions, mode, target>;	\
	public:																	\
		accessor(buffer<DataType, dimensions>& bufferRef)					\
			: Base(bufferRef) {}											\
		accessor(															\
			buffer<DataType, dimensions>& bufferRef,						\
			range<dimensions> offset,										\
			range<dimensions> range											\
		)																	\
			: Base(bufferRef, offset, range) {}								\
	};
#else
#define SYCL_ADD_ACCESSOR_BUFFER(mode)										\
	SYCL_ADD_ACCESSOR(mode) {												\
		using Base = detail::accessor_<DataType, dimensions, mode, target>;	\
	public:																	\
		using Base::accessor_;												\
	};
#endif

SYCL_ADD_ACCESSOR_BUFFER(access::read)
SYCL_ADD_ACCESSOR_BUFFER(access::write)
SYCL_ADD_ACCESSOR_BUFFER(access::read_write)
SYCL_ADD_ACCESSOR_BUFFER(access::discard_read_write)

#undef SYCL_ADD_ACCESSOR_BUFFER

} // namespace sycl
} // namespace cl

#undef SYCL_ACCESSOR_DEVICE_REF_CONSTRUCTOR
#undef SYCL_DEVICE_REF_SUBSCRIPT_OP
#undef SYCL_DEVICE_REF_SUBSCRIPT_OPERATORS
#undef SYCL_ACCESSOR_HOST_REF_CONSTRUCTOR
#undef SYCL_BUFFER_CONSTRUCTORS

#undef SYCL_ACCESSOR_CLASS
#undef SYCL_ADD_ACCESSOR
