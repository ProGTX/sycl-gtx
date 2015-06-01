#pragma once

// Device buffer accessors
// 3.4.6 Accessors and 3.4.6.4 Buffer accessors

#include "buffer_base.h"
#include "device_reference.h"
#include "../access.h"
#include "../accessor.h"
#include "../ranges/id.h"
#include "../../common.h"
#include "../../data_ref.h"

namespace cl {
namespace sycl {
namespace detail {

SYCL_ACCESSOR_CLASS(
	target == access::constant_buffer	||
	target == access::global_buffer
),
	public accessor_buffer<DataType, dimensions>,
	public accessor_device_ref<dimensions, DataType, dimensions, (access::mode)mode, (access::target)target>
{
	template <int level, typename DataType, int dimensions, access::mode mode, access::target target>
	friend class accessor_device_ref;

public:
	accessor_(
		cl::sycl::buffer<DataType, dimensions>& bufferRef,
		handler* commandGroupHandler,
		range<dimensions> offset,
		range<dimensions> range
	)	:	accessor_buffer(bufferRef, commandGroupHandler, offset, range),
			accessor_device_ref(this, {})
	{}
	accessor_(cl::sycl::buffer<DataType, dimensions>& bufferRef, handler* commandGroupHandler)
		: accessor_(
			bufferRef,
			commandGroupHandler,
			detail::empty_range<dimensions>(),
			bufferRef.get_range()
		) {}
	accessor_(const accessor_& copy)
		:	accessor_buffer((const accessor_buffer<DataType, dimensions>&)copy),
			accessor_device_ref(this, copy)
	{}
	accessor_(accessor_&& move)
		:	accessor_buffer(std::move((accessor_buffer<DataType, dimensions>)move)),
			accessor_device_ref(
				this,
				std::move(
					(accessor_device_ref<dimensions, DataType, dimensions, (access::mode)mode, (access::target)target>)move
				)
			)
	{}

	virtual cl_mem get_cl_mem_object() const override {
		return get_buffer_object();
	}

	data_ref operator[](id<dimensions> index) const {
		auto resource_name = kernel_::source::register_resource(*this);
		return data_ref(
			resource_name + "[" + data_ref::get_name(index) + "]"
		);
	}

private:
	using subscript_return_t = typename subscript_helper<dimensions, DataType, dimensions, (access::mode)mode, (access::target)target>::type;
public:
	SYCL_DEVICE_REF_SUBSCRIPT_OPERATORS();

protected:
	virtual void* resource() const override {
		return buf;
	}

	virtual size_t argument_size() const override {
		return sizeof(cl_mem);
	}
};

} // namespace detail
} // namespace sycl
} // namespace cl
