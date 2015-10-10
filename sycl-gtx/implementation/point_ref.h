#pragma once

#include "common.h"
#include "data_ref.h"
#include "ptr_or_val.h"

namespace cl {
namespace sycl {
namespace detail {

template <bool is_const, typename data_basic_t = size_t>
struct point_ref : data_ref {
protected:
	template <typename T>
	struct is_computable {
		static const bool value = !is_const && std::is_arithmetic<T>::value;
	};
	template <typename T>
	using if_is_num_assignable = typename std::enable_if<is_computable<T>::value>::type;
	template <typename T>
	using if_is_numeric = typename std::enable_if<std::is_arithmetic<T>::value>::type;

	using data_ptr_t = typename std::conditional<is_const, data_basic_t* const, data_basic_t*>::type;
	using data_t = typename std::remove_pointer<data_ptr_t>::type;

	// TODO: Need to also carry references to type and name
	ptr_or_val<data_t> data;

public:
	point_ref(data_basic_t& data, string_class name, type_t type_)
		: data_ref(name), data(&data) {
		type = type_;
	}
	point_ref(data_basic_t value, type_t type_, bool)
		: data_ref(std::to_string(value)), data(nullptr, value) {
		type = type_;
	}
	point_ref(string_class name, type_t type_, bool)
		: data_ref(name), data(nullptr, 0) {
		type = type_;
	}

	operator data_basic_t() const {
		return data;
	}
	template <class = typename std::enable_if<!is_const>::type>
	operator data_basic_t&() {
		return data;
	}

	template <typename T, class = if_is_num_assignable<T>>
	point_ref& operator=(T n) {
		if(type == type_t::numeric) {
			data = n;
			name = std::to_string(data);
		}
		else {
			data_ref::operator=(n);
		}
		return *this;
	}
	template <typename T, class = if_is_num_assignable<T>>
	point_ref& operator+=(T n) {
		if(type == type_t::numeric) {
			data += n;
			name = std::to_string(data);
		}
		else {
			data_ref::operator+=(n);
		}
		return *this;
	}

	template <typename T, class = if_is_numeric<T>>
	point_ref operator*(T n) const {
		if(type == type_t::numeric) {
			return point_ref(data * n, type, true);
		}
		else {
			auto ret = data_ref::operator*(n);
			return point_ref(std::move(ret.name), ret.type, true);
		}
	}
	template <typename T, class = if_is_num_assignable<T>>
	friend point_ref operator-(T n, const point_ref& rhs) {
		if(type == type_t::numeric) {
			return point_ref(n - data, type, true);
		}
		else {
			auto ret = data_ref::operator-(n, *this);
			return point_ref(std::move(ret.name), ret.type, true);
		}
	}

	// TODO: Other operators

	// TODO: enable_if causes here an internal MSVC error C1001
	// TODO: data_ref::operator&
	//template <class = typename std::enable_if<!is_const>::type>
	point_ref<is_const, data_basic_t*> operator&() {
		string_class name_;
		if(type == type_t::numeric) {
			name_ = name;
		}
		else {
			name_ = string_class("&(") + name + ")";
		}

		return point_ref<is_const, data_basic_t*>(&data, name_, type);
	}

	// TODO: enable_if causes here an internal MSVC error C1001
	// TODO: data_ref::operator*
	//template <class = typename std::enable_if<std::is_pointer<data_basic_t>::value>::type>
	point_ref<is_const, typename std::remove_pointer<data_basic_t>::type> operator*() {
		string_class name_;
		if(type == type_t::numeric) {
			name_ = name;
		}
		else {
			name_ = string_class("*(") + name + ")";
		}

		return point_ref<is_const, typename std::remove_pointer<data_basic_t>::type>(*data, name_, type);
	}
};

} // namespace detail
} // namespace sycl
} // namespace cl
