#pragma once

#include "kernel_source.h"
#include "../specification/ranges.h"
#include "../common.h"
#include "../debug.h"

namespace cl {
namespace sycl {

namespace detail {

using kernel_::source;

template <int dimensions>
struct generate_id_code {
	static void global() {
		for(int i = 0; i < dimensions; ++i) {
			auto id_s = std::to_string(i);
			source::add(
				string_class("const int ") + point_names::id_global + id_s +
				" = get_global_id(" + id_s + ")"
			);
		}

		if(dimensions == 2) {
			source::add(
				string_class("const int ") + point_names::id_global +
				" = " + point_names::id_global + "1 * get_global_size(0) + " + point_names::id_global + "0"
			);
		}

		// TODO: 3d
	}

	static void local() {
		for(int i = 0; i < dimensions; ++i) {
			auto id_s = std::to_string(i);
			source::add(
				string_class("const int ") + point_names::id_local + id_s +
				" = get_local_id(" + id_s + ")"
			);
		}

		if(dimensions == 2) {
			source::add(
				string_class("const int ") + point_names::id_local +
				" = " + point_names::id_local + "1 * get_local_size(0) + " + point_names::id_local + "0"
			);
		}

		// TODO: 3d
	}
};

namespace kernel_ {

template <class Input>
struct constructor;

// Single task invoke
template <>
struct constructor<void> {
	static source get(function_class<void(void)> kern) {
		source src;
		source::enter(src);

		kern();

		return source::exit(src);
	}
};

// Parallel For with range and kernel parameter id
template <int dimensions>
struct constructor<id<dimensions>> {
	static id<dimensions> global_id() {
		auto i = id<dimensions>();
		i.set(data_ref::type_t::id_global);
		return i;
	}

	static source get(function_class<void(id<dimensions>)> kern) {
		source src;
		source::enter(src);

		// TODO: num_work_items, work_item_offset
		generate_id_code<dimensions>::global();
		kern(global_id());

		return source::exit(src);
	}
};

// Parallel For with range and kernel parameter item
template <int dimensions>
struct constructor<item<dimensions>> {
	static source get(function_class<void(item<dimensions>)> kern) {
		source src;
		source::enter(src);

		generate_id_code<dimensions>::global();
		auto index = constructor<id<dimensions>>::global_id();
		// TODO: num_work_items, work_item_offset
		//item<dimensions> it(index, num_work_items, work_item_offset);
		item<dimensions> it(index, detail::empty_range<dimensions>());
		kern(it);

		return source::exit(src);
	}
};

// Parallel For with nd_range
template <int dimensions>
struct constructor<nd_item<dimensions>> {
	static id<dimensions> local_id() {
		auto i = id<dimensions>();
		i.set(data_ref::type_t::id_local);
		return i;
	}

	static source get(function_class<void(nd_item<dimensions>)> kern) {
		source src;
		source::enter(src);

		// TODO: execution_range
		auto r = detail::empty_range<dimensions>();
		nd_range<dimensions> execution_range(r, r);

		generate_id_code<dimensions>::global();
		auto global_id = constructor<id<dimensions>>::global_id();

		item<dimensions> global_item(
			global_id,
			execution_range.get_global(),
			execution_range.get_offset()
		);

		generate_id_code<dimensions>::local();

		// TODO: Store group ID into offset of local_item
		item<dimensions> local_item(
			local_id(),
			execution_range.get_local(),
			execution_range.get_offset()
		);

		nd_item<dimensions> it(std::move(global_item), std::move(local_item));
		kern(it);

		return source::exit(src);
	}
};

} // namespace kernel_
} // namespace detail

} // namespace sycl
} // namespace cl
