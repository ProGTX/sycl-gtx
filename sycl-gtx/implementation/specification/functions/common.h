#pragma once

// 3.9.5 Common Functions

#include "../types.h"
#include "../../data_ref.h"

namespace cl {
namespace sycl {

template <class First, class Second>
static detail::data_ref min(First first, Second second) {
	using detail::data_ref;
	return data_ref(string_class("min(") + data_ref::get_name(first) + ", " + data_ref::get_name(second) + ")");
}

} // namespace sycl
} // namespace cl
