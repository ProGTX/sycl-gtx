#pragma once

#include "gen_source.h"
#include "specification\ranges.h"
#include "common.h"
#include "debug.h"

namespace cl {
namespace sycl {

namespace detail {
namespace kernel_ {

template<class Input>
struct constructor;

// Single task invoke
template<>
struct constructor<void> {
	static source get(function_class<void> kern) {
		source src;
		source::scope = &src;

		kern(); // MSVC2013 complains about this, but compiles and links.

		source::scope = nullptr;
		return src;
	}
};

// Parallel For with range and kernel parameter id
template <int dimensions>
struct constructor<id<dimensions>> {
	static id<dimensions> generate_global_id_code(range<dimensions>& num_work_items) {
		for(int i = 0; i < dimensions; ++i) {
			auto id_s = std::to_string(i);
			source::add(
				string_class("const int ") + id_global_name + id_s +
				" = get_global_id(" + id_s + ")"
			);
		}

		if(dimensions == 2) {
			source::add(
				string_class("const int ") + id_global_all_name + " = " +
				id_global_name + "1 * " + std::to_string(num_work_items[0]) + " + " + id_global_name + "0"
			);
		}

		// TODO: 3d

		return id<dimensions>{0, 0, 0};
	}

	static source get(function_class<id<dimensions>> kern, range<dimensions>& num_work_items, id<dimensions>& work_item_offset) {
		source src;
		source::scope = &src;

		auto index = generate_global_id_code(num_work_items);
		kern(index);

		source::scope = nullptr;
		return src;
	}
};

// Parallel For with range and kernel parameter item
template <int dimensions>
struct constructor<item<dimensions>> {
	static source get(function_class<item<dimensions>> kern, range<dimensions>& num_work_items, id<dimensions>& work_item_offset) {
		source src;
		source::scope = &src;

		auto index = constructor<id<dimensions>>::generate_global_id_code(num_work_items);
		item<dimensions> it(index, num_work_items, work_item_offset);
		kern(it);

		source::scope = nullptr;
		return src;
	}
};

// Parallel For with nd_range
template <int dimensions>
struct constructor<nd_item<dimensions>> {
	static id<dimensions> generate_local_id_code() {
		for(int i = 0; i < dimensions; ++i) {
			auto id_s = std::to_string(i);
			source::add(
				string_class("const int ") + id_local_name + id_s +
				" = get_local_id(" + id_s + ")"
			);
		}

		// TODO: 2d and 3d

		auto i = id<dimensions>{0, 0, 0};
		i.type = id_ref::type::local;
		return i;
	}

	static source get(function_class<nd_item<dimensions>> kern, nd_range<dimensions>& execution_range) {
		source src;
		source::scope = &src;

		item<dimensions> global_item(
			constructor<id<dimensions>>::generate_global_id_code(execution_range.get_global_range()),
			execution_range.get_global_range(),
			execution_range.get_offset()
		);

		// TODO: Store group ID into offset of local_item
		item<dimensions> local_item(
			generate_local_id_code(),
			execution_range.get_local_range(),
			execution_range.get_offset()
		);

		nd_item<dimensions> it(std::move(global_item), std::move(local_item));
		kern(it);

		source::scope = nullptr;
		return src;
	}
};

} // namespace kernel_
} // namespace detail

} // namespace sycl
} // namespace cl
