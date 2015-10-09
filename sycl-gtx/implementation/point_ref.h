#pragma once

#include "data_ref.h"
#include "common.h"

namespace cl {
namespace sycl {
namespace detail {

template <typename T>
struct ptr_or_val {
private:
	bool is_owner;
	T* data;

public:
	ptr_or_val(nullptr_t, T value)
		: is_owner(true), data(reinterpret_cast<T*>(value)) {}
	ptr_or_val(T* ptr)
		: is_owner(false), data(ptr) {}

	operator T() const {
		if(is_owner) {
			return reinterpret_cast<T>(data);
		}
		else {
			return *data;
		}
	}
	operator T&() {
		if(is_owner) {
			return reinterpret_cast<T&>(data);
		}
		else {
			return *data;
		}
	}

	ptr_or_val& operator=(T n) {
		if(is_owner) {
			data = reinterpret_cast<T*>(n);
		}
		else {
			*data = n;
		}
		return *this;
	}
	ptr_or_val<T*> operator&() {
		return ptr_or_val<T*>(&data);
	}
};

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
			auto lhs = data_ref::operator*(n);
			return point_ref(std::move(lhs.name), lhs.type, true);
		}
	}

	// TODO: enable_if causes here an internal MSVC error C1001
	//template <class = typename std::enable_if<!is_const>::type>
	ptr_or_val<data_ptr_t> operator&() {
		return &(data);
	}
};

} // namespace detail
} // namespace sycl
} // namespace cl
