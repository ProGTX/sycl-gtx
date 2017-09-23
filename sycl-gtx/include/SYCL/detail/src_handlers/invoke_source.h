#pragma once

#include "SYCL/detail/common.h"
#include "SYCL/detail/debug.h"
#include "SYCL/detail/src_handlers/kernel_source.h"
#include "SYCL/ranges.h"

namespace cl {
namespace sycl {

namespace detail {

using kernel_ns::source;

template <int dimensions, bool is_id>
struct identifier_code {
  static string_class get_function_name(
      typename point<dimensions>::type_t type) {
    using type_t = typename point<dimensions>::type_t;
    string_class name = "";
    switch (type) {
      case type_t::id_global:
        name = "get_global_id";
        break;
      case type_t::id_local:
        name = "get_local_id";
        break;
      case type_t::range_global:
        name = "get_global_size";
        break;
      case type_t::range_local:
        name = "get_local_size";
        break;
      default:
        break;
    }
    return name;
  }
  static void generate(typename point<dimensions>::type_t type) {
    string_class name = point<dimensions>::name_from_type(type);
    string_class function_name = get_function_name(type);

    for (int i = 0; i < dimensions; ++i) {
      auto id_s = get_string<int>::get(i);
      source::add(string_class("const int ") + name + id_s + " = " +
                  function_name + "(" + id_s + ")");
    }

    if (is_id) {
      string_replace_one(function_name, "id", "size");

      if (dimensions == 1) {
        source::add(string_class("const int ") + name + " = " + name + "0");
      }
      if (dimensions == 2) {
        source::add(string_class("const int ") + name + " = " + name + "1 * " +
                    function_name + "(0) + " + name + "0");
      }

      // TODO(progtx): 3d
    }
  }
};

template <int dimensions>
struct generate_id_refs {
  static void global() {
    identifier_code<dimensions, true>::generate(
        point<dimensions>::type_t::id_global);
  }
  static void local() {
    identifier_code<dimensions, true>::generate(
        point<dimensions>::type_t::id_local);
  }
};

template <int dimensions>
struct generate_range_refs {
  static void global() {
    identifier_code<dimensions, true>::generate(
        point<dimensions>::type_t::range_global);
  }
  static void local() {
    identifier_code<dimensions, true>::generate(
        point<dimensions>::type_t::range_local);
  }
};

namespace kernel_ns {

template <class Input>
struct constructor;

/**
 * Single task invoke
 */
template <>
struct constructor<void> {
  static source get(function_class<void(void)> kern) {
    source src;
    source::enter(src);

    kern();

    return source::exit(src);
  }
};

/**
 * Parallel For with range and kernel parameter id
 */
template <int dimensions>
struct constructor<id<dimensions>> {
  static source get(function_class<void(id<dimensions>)> kern) {
    source src;
    source::enter(src);

    // TODO(progtx): num_work_items, work_item_offset
    generate_id_refs<dimensions>::global();
    kern(get_special_id<dimensions>::global());

    return source::exit(src);
  }
};

/**
 * Parallel For with range and kernel parameter item
 */
template <int dimensions>
struct constructor<item<dimensions>> {
  static source get(function_class<void(item<dimensions>)> kern) {
    source src;
    source::enter(src);

    generate_id_refs<dimensions>::global();
    auto index = get_special_id<dimensions>::global();
    // TODO(progtx): num_work_items, work_item_offset
    // item<dimensions> it(index, num_work_items, work_item_offset);
    item<dimensions> it(index, empty_range<dimensions>());
    kern(it);

    return source::exit(src);
  }
};

/**
 * Parallel For with nd_range
 */
template <int dimensions>
struct constructor<nd_item<dimensions>> {
  static source get(function_class<void(nd_item<dimensions>)> kern) {
    source src;
    source::enter(src);

    generate_id_refs<dimensions>::global();
    generate_id_refs<dimensions>::local();
    generate_range_refs<dimensions>::global();
    generate_range_refs<dimensions>::local();

    auto grange = get_special_range<dimensions>::global();
    auto lrange = get_special_range<dimensions>::local();
    nd_range<dimensions> execution_range(grange, lrange);

    auto global_id = get_special_id<dimensions>::global();

    item<dimensions> global_item(global_id, execution_range.get_global(),
                                 execution_range.get_offset());

    // TODO(progtx): Store group ID into offset of local_item
    item<dimensions> local_item(get_special_id<dimensions>::local(),
                                execution_range.get_local(),
                                execution_range.get_offset());

    nd_item<dimensions> it(std::move(global_item), std::move(local_item));
    kern(it);

    return source::exit(src);
  }
};

}  // namespace kernel_ns
}  // namespace detail

}  // namespace sycl
}  // namespace cl
