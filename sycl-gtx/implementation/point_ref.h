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
// TODO: Consider specialization depending on ownership of data
template <bool is_const>
struct shared_point_data : shared_ptr_class<size_t> {
private:
	using uint = typename shared_data_base<is_const>::uint;
	using Base = shared_ptr_class<uint>;
public:
	shared_point_data(uint value, bool)
		: Base(new uint(value)) {}
	shared_point_data(uint* ptr)
		: Base(ptr, [](uint* p) { /* Don't manage borrowed pointer */ }) {}
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
		: data_ref(std::to_string(value)), data(value, true) {
		type = type_;
	}
	point_ref(string_class name, type_t type_, bool)
		: data_ref(name), data(0, true) {
		type = type_;
	}

	operator size_t() const {
		return *data;
	}
	template <class = typename std::enable_if<!is_const>::type>
	operator size_t&() {
		return *data;
	}

	template <typename T, class = if_is_num_assignable<T>>
	point_ref& operator=(T n) {
		if(type == type_t::numeric) {
			*data = n;
			name = std::to_string(*data);
		}
		else {
			data_ref::operator=(n);
		}
		return *this;
	}
	template <typename T, class = if_is_num_assignable<T>>
	point_ref& operator+=(T n) {
		if(type == type_t::numeric) {
			*data += n;
			name = std::to_string(*data);
		}
		else {
			data_ref::operator+=(n);
		}
		return *this;
	}

	template <typename T, class = if_is_numeric<T>>
	point_ref operator*(T n) const {
		if(type == type_t::numeric) {
			return point_ref((*data) * n, type, true);
		}
		else {
			auto lhs = data_ref::operator*(n);
			return point_ref(std::move(lhs.name), lhs.type, true);
		}
	}

	typename shared_data_base<is_const>::uint_ptr operator&() {
		return data.get();
	}
};

} // namespace detail
} // namespace sycl
} // namespace cl
