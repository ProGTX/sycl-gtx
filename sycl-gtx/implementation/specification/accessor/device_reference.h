#pragma once

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

template <typename DataType>
struct acc_device_return {
	using type = data_ref;
};

template <int level, typename DataType, int dimensions, access::mode mode, access::target target>
struct subscript_helper {
	using type = accessor_device_ref<level - 1, DataType, dimensions, mode, target>;
};
template <typename DataType, int dimensions, access::mode mode, access::target target>
struct subscript_helper<1, DataType, dimensions, mode, target> {
	using type = typename acc_device_return<DataType>::type;
};

#define SYCL_ACCESSOR_DEVICE_REF_CONSTRUCTOR()									\
	using acc_t = accessor_<DataType, dimensions, mode, target>;				\
	friend class acc_t;															\
	friend class accessor_device_ref;											\
	const acc_t* parent;														\
	vector_class<string_class> rang;											\
	accessor_device_ref(const acc_t* parent, vector_class<string_class> range)	\
		: parent(parent), rang(range) {											\
		rang.resize(3);															\
	}																			\
	accessor_device_ref(const acc_t* parent, const accessor_device_ref& copy)	\
		: parent(parent), rang(copy.rang) {}									\
	accessor_device_ref(const acc_t* parent, accessor_device_ref&& move)		\
		: parent(parent), rang(std::move(move.rang)) {}							\
	friend void swap(accessor_device_ref& first, accessor_device_ref& second) {	\
		std::swap(first.rang, second.rang);										\
	}

#define SYCL_DEVICE_REF_SUBSCRIPT_OP(type)						\
	subscript_return_t operator[](const type& index) const {	\
		return subscript(index);								\
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
	subscript_return_t subscript(const T& index) const {
		auto rang_copy = rang;
		rang_copy[dimensions - level] = data_ref::get_name(index);
		return subscript_return_t(parent, rang_copy);
	}
public:
	SYCL_DEVICE_REF_SUBSCRIPT_OPERATORS();
};

template <typename DataType, int dimensions, access::mode mode, access::target target>
class accessor_device_ref<1, DataType, dimensions, mode, target> {
protected:
	using subscript_return_t = typename acc_device_return<DataType>::type;
	SYCL_ACCESSOR_DEVICE_REF_CONSTRUCTOR();

	template <class T>
	subscript_return_t subscript(const T& index) const {
		// Basically the same as with host buffer accessor, just dealing with strings
		auto rang_copy = rang;
		rang_copy[dimensions - 1] = data_ref::get_name(index);
		string_class ind(std::move(rang_copy[0]));
		auto multiplier = parent->access_buffer_range(0);
		for(int i = 1; i < dimensions; ++i) {
			ind += string_class(" + ") + std::move(rang_copy[i]) + " * " + get_string(multiplier);
			multiplier *= parent->access_buffer_range(i);
		}
		auto resource_name = kernel_::source::register_resource(*parent);
		return subscript_return_t(
			resource_name + "[" + ind + "]"
		);
	}
public:
	SYCL_DEVICE_REF_SUBSCRIPT_OPERATORS();
};

} // namespace detail
} // namespace sycl
} // namespace cl

#undef SYCL_ACCESSOR_DEVICE_REF_CONSTRUCTOR
