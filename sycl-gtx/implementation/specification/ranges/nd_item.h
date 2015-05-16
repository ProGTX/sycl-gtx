#pragma once

// 3.7.1.5 nd_item class

#include "../access.h"
#include "../../data_ref.h"

namespace cl {
namespace sycl {

// Forward declarations
template <int dims>
struct id;
template <int dims>
struct range;
template <int dims>
struct nd_range;

namespace detail {
namespace kernel_ {
	template <class Input>
	struct constructor;
}
}

template <int dims = 1>
struct nd_item : public item<dims> {
public:
	// Remains of the item class
	size_t get(int dimension) const = delete;
	size_t& operator[](int dimension) = delete;

protected:
	friend struct detail::kernel_::constructor<nd_item<dims>>;

	item<dims> local_item;

	nd_item(item<dims> global_item, item<dims> local_item)
		: item(global_item), local_item(local_item) {}

	// A bit of a hack - to the outside it appears to conform to the specification
	using size_t = detail::id_ref;

public:
	id<dims> get_global_id() const {
		return item::get_global_id();
	}
	size_t get_global_id(int dimension) const {
		return get_global_id()[dimension];
	}

	id<dims> get_local_id() const {
		return local_item.get_global_id();
	}
	size_t get_local_id(int dimension) const {
		return get_local_id()[dimension];
	}

	id<dims> get_group_id() const {
		return local_item.get_offset();
	}
	size_t get_group_id(int dimension) const {
		return get_group_id()[dimension];
	}

	range<dims> get_global_range() const {
		return item::get_global_range();
	}
	range<dims> get_local_range() const {
		return local_item.get_global_range();
	}
	id<dims> get_offset() const {
		return item::get_offset();
	}
	nd_range<dims> get_nd_range() const {
		return nd_range<dims>(get_global_range(), get_local_range(), get_offset());
	}

	void barrier(access::fence_space flag = access::fence_space::global_and_local) const {
		string_class flag_string;

		switch(flag) {
			case access::fence_space::local:
				flag_string = "CLK_LOCAL_MEM_FENCE";
				break;
			case access::fence_space::global:
				flag_string = "CLK_GLOBAL_MEM_FENCE";
				break;
			case access::fence_space::global_and_local:
			default:
				flag_string = "CLK_LOCAL_MEM_FENCE|CLK_GLOBAL_MEM_FENCE";
				break;
		}

		detail::kernel_add(string_class("barrier(") + flag_string + ")");
	}
};

} // namespace sycl
} // namespace cl
