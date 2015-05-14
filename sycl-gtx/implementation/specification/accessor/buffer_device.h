#pragma once

// Device buffer accessors

#include "buffer_base.h"
#include "../access.h"
#include "../accessor.h"
#include "../ranges/id.h"
#include "../../common.h"
#include "../../data_ref.h"

namespace cl {
namespace sycl {
namespace detail {

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
	template <class T>
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
	template <class T>
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
	accessor_(
		cl::sycl::buffer<DataType, dimensions>& bufferRef,
		range<dimensions> offset,
		range<dimensions> range
	) : accessor_buffer(bufferRef, offset, range) {
		acc = this;
	}
	accessor_(cl::sycl::buffer<DataType, dimensions>& bufferRef)
		: accessor_(
		bufferRef,
		detail::empty_range<dimensions>(),
		bufferRef.get_range()
	) {}

	virtual cl_mem get_cl_mem_object() const override {
		return get_buffer_object();
	}

	data_ref operator[](id<dimensions> index) const {
		kernel_::source::register_resource(*this);
		return data_ref(
			get_resource_name() + "[" + data_ref::get_name(index) + "]"
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

} // namespace detail
} // namespace sycl
} // namespace cl

#undef SYCL_ACCESSOR_DEVICE_REF_CONSTRUCTOR
#undef SYCL_DEVICE_REF_SUBSCRIPT_OP
#undef SYCL_DEVICE_REF_SUBSCRIPT_OPERATORS
