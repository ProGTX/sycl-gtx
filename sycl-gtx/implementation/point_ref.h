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

template <bool is_const>
struct shared_data_base {
	using uint_ptr = typename std::conditional<is_const, size_t* const, size_t*>::type;
	using uint = typename std::remove_pointer<uint_ptr>::type;
};

// TODO: Also capture type
template <bool is_const>
struct shared_point_data : shared_data_base<is_const> {
public:
	ptr_or_val<uint> value;

	shared_point_data(nullptr_t, uint val)
		: value(nullptr, val) {}
	shared_point_data(uint* ptr)
		: value(ptr) {}
};

template <bool is_const>
struct point_ref : data_ref {
protected:
	using data_t = shared_point_data<is_const>;
	data_t data;

	template <typename T>
	struct is_computable {
		static const bool value = !is_const && std::is_arithmetic<T>::value;
	};
	template <typename T>
	using if_is_num_assignable = typename std::enable_if<is_computable<T>::value>::type;
	template <typename T>
	using if_is_numeric = typename std::enable_if<std::is_arithmetic<T>::value>::type;

public:
	point_ref(size_t& data, string_class name, type_t type_)
		: data_ref(name), data(&data) {
		type = type_;
	}
	point_ref(size_t value, type_t type_, bool)
		: data_ref(std::to_string(value)), data(nullptr, value) {
		type = type_;
	}
	point_ref(string_class name, type_t type_, bool)
		: data_ref(name), data(nullptr, 0) {
		type = type_;
	}

	operator size_t() const {
		return data.value;
	}
	template <class = typename std::enable_if<!is_const>::type>
	operator size_t&() {
		return data.value;
	}

	template <typename T, class = if_is_num_assignable<T>>
	point_ref& operator=(T n) {
		if(type == type_t::numeric) {
			data.value = n;
			name = std::to_string(data.value);
		}
		else {
			data_ref::operator=(n);
		}
		return *this;
	}
	template <typename T, class = if_is_num_assignable<T>>
	point_ref& operator+=(T n) {
		if(type == type_t::numeric) {
			data.value += n;
			name = std::to_string(data.value);
		}
		else {
			data_ref::operator+=(n);
		}
		return *this;
	}

	template <typename T, class = if_is_numeric<T>>
	point_ref operator*(T n) const {
		if(type == type_t::numeric) {
			return point_ref(data.value * n, type, true);
		}
		else {
			auto lhs = data_ref::operator*(n);
			return point_ref(std::move(lhs.name), lhs.type, true);
		}
	}

	// TODO: enable_if causes here an internal MSVC error C1001
	//template <class = typename std::enable_if<!is_const>::type>
	ptr_or_val<typename shared_data_base<is_const>::uint_ptr> operator&() {
		return &(data.value);
	}
};

} // namespace detail
} // namespace sycl
} // namespace cl
