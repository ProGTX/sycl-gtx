#pragma once

// Host buffer accessors
// 3.6.4.7 Host accessors

#include "buffer_base.h"
#include "../access.h"
#include "../accessor.h"
#include "../ranges.h"
#include <array>

namespace cl {
namespace sycl {
namespace detail {

#define SYCL_ACCESSOR_HOST_REF_CONSTRUCTOR()									\
	using acc_t = accessor_<DataType, dimensions, mode, access::host_buffer>;	\
	friend class acc_t;															\
	friend class accessor_host_ref;												\
	const acc_t* parent;														\
	std::array<size_t, 3> rang;													\
	accessor_host_ref(const acc_t* parent, std::array<size_t, 3> rang)			\
		: parent(parent), rang(rang) {}											\
	accessor_host_ref(const acc_t* parent, const accessor_host_ref& copy)		\
		: parent(parent), rang(copy.rang) {}									\
	accessor_host_ref(const acc_t* parent, accessor_host_ref&& move)			\
		: parent(parent), rang(std::move(move.rang)) {}							\
	friend void swap(accessor_host_ref& first, accessor_host_ref& second) {		\
		std::swap(first.rang, second.rang);										\
	}

template <int level, typename DataType, int dimensions, access::mode mode>
class accessor_host_ref {
protected:
	using Lower = accessor_host_ref<dimensions - 1, DataType, dimensions, mode>;
	SYCL_ACCESSOR_HOST_REF_CONSTRUCTOR();
public:
	Lower operator[](int index) {
		auto rang_copy = rang;
		rang_copy[dimensions - level] = index;
		return Lower(parent, rang_copy);
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
			multiplier *= parent->access_buffer_range(i);
		}
		return parent->access_host_data()[index];
	}
};

SYCL_ACCESSOR_CLASS(target == access::host_buffer),
	public accessor_buffer<DataType, dimensions>,
	public accessor_host_ref<dimensions, DataType, dimensions, (access::mode)mode>
{
	template <int level, typename DataType, int dimensions, access::mode mode>
	friend class accessor_host_ref;
public:
	accessor_(
		buffer<DataType, dimensions>& bufferRef,
		range<dimensions> offset,
		range<dimensions> range
	)	:	accessor_buffer(bufferRef, nullptr, offset, range),
			accessor_host_ref(this, std::array<size_t, 3> { 0, 0, 0 })
	{}
	accessor_(buffer<DataType, dimensions>& bufferRef)
		: accessor_(
			bufferRef,
			detail::empty_range<dimensions>(),
			bufferRef.get_range()
		) {}
	accessor_(const accessor_& copy)
		:	accessor_buffer((const accessor_buffer<DataType, dimensions>&)copy),
			accessor_host_ref(this, copy)
	{}
	accessor_(accessor_&& move)
		:	accessor_buffer(std::move((accessor_buffer<DataType, dimensions>)move)),
			accessor_host_ref(
				this,
				std::move(
					(accessor_host_ref<dimensions, DataType, dimensions, (access::mode)mode>)move
				)
			)
	{}
};

} // namespace detail
} // namespace sycl
} // namespace cl

#undef SYCL_ACCESSOR_HOST_REF_CONSTRUCTOR
