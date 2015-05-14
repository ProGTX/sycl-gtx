#pragma once

// Host buffer accessors
// 3.6.4.7 Host accessors

#include "buffer_base.h"
#include "../access.h"
#include "../accessor.h"
#include "../ranges.h"

namespace cl {
namespace sycl {
namespace detail {

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
	accessor_(
		buffer<DataType, dimensions>& bufferRef,
		range<dimensions> offset,
		range<dimensions> range
	) : accessor_buffer(bufferRef, offset, range) {
		acc = this;
	}
	accessor_(buffer<DataType, dimensions>& bufferRef)
		: accessor_(
		bufferRef,
		detail::empty_range<dimensions>(),
		bufferRef.get_range()
	) {}
};

} // namespace detail
} // namespace sycl
} // namespace cl

#undef SYCL_ACCESSOR_HOST_REF_CONSTRUCTOR
