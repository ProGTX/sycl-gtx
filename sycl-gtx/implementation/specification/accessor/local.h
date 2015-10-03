#pragma once

// 3.6.4.6 Local accessors

#include "device_reference.h"
#include "../access.h"
#include "../accessor.h"
#include "../buffer.h"
#include "../command_group.h"
#include "../ranges.h"
#include "../../counter.h"

namespace cl {
namespace sycl {

namespace detail {

SYCL_ACCESSOR_CLASS(target == access::local),
	protected counter<accessor_<DataType, dimensions, mode, target>>,
	public accessor_device_ref<dimensions, DataType, dimensions, (access::mode)mode, (access::target)target>
{
protected:
	template <int level, typename DataType, int dimensions, access::mode mode, access::target target>
	friend class accessor_device_ref;

	range<dimensions> allocationSize;

	size_t access_buffer_range(int n) const {
		return allocationSize.get(n);
	}

	virtual void* resource() const override {
		return reinterpret_cast<void*>(get_count_id());
	}

	virtual size_t argument_size() const override {
		return sizeof(DataType) * detail::get_size(allocationSize);
	}

public:
	accessor_(range<dimensions> allocationSize)
		:	accessor_device_ref(this, {}),
			allocationSize(allocationSize) {
		// TODO
		if(command::group_::in_scope()) {
			command::group_::add(
				buffer_access{ nullptr, (access::mode)mode, (access::target)target },
				__func__
			);
		}
	}

private:
	using subscript_return_t = typename subscript_helper<dimensions, DataType, dimensions, (access::mode)mode, (access::target)target>::type;
public:
	SYCL_DEVICE_REF_SUBSCRIPT_OPERATORS();
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
