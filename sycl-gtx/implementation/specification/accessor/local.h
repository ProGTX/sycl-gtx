#pragma once

// 3.6.4.6 Local accessors

#include "../access.h"
#include "../accessor.h"
#include "../buffer.h"
#include "../command_group.h"
#include "../ranges.h"

namespace cl {
namespace sycl {

namespace detail {

SYCL_ACCESSOR_CLASS(target == access::local) {
	accessor_(range<dimensions> allocationSize) {
		// TODO
		if(command::group_::in_scope()) {
			command::group_::add(buffer_access{ nullptr, mode, target }, __func__);
		}
	}
};

} // namespace detail

#if MSVC_LOW
#define SYCL_ADD_ACCESSOR_LOCAL(mode)												\
	SYCL_ADD_ACCESSOR(mode, access::local) {										\
		using Base = detail::accessor_<DataType, dimensions, mode, access::local>;	\
	public:																			\
		accessor(range<dimensions> allocationSize)									\
			: Base(allocationSize) {}												\
	};
#else
#define SYCL_ADD_ACCESSOR_LOCAL(mode)												\
	SYCL_ADD_ACCESSOR(mode, access::local) {										\
		using Base = detail::accessor_<DataType, dimensions, mode, access::local>;	\
	public:																			\
		using Base::accessor_;														\
	};
#endif

// 3.6.4.9 Accessor capabilities and restrictions
SYCL_ADD_ACCESSOR_LOCAL(access::read)
SYCL_ADD_ACCESSOR_LOCAL(access::write)
SYCL_ADD_ACCESSOR_LOCAL(access::read_write)

} // namespace sycl
} // namespace cl
