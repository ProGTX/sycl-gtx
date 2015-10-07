#pragma once

#include "../../data_ref.h"
#include "../../common.h"

namespace cl {
namespace sycl {
namespace detail {

template <bool is_const>
struct shared_data {
	using ptr_t = typename std::conditional<is_const, size_t* const, size_t*>::type;
	using type = typename std::remove_pointer<ptr_t>::type;
};

template <bool is_const>
struct shared_uint : shared_ptr_class<size_t> {
private:
	using type = typename shared_data<is_const>::type;
	using Base = shared_ptr_class<type>;
public:
	shared_uint(size_t value, bool)
		: Base(new type(value)) {}
	shared_uint(type* ptr)
		: Base(ptr, [](type* p) { /* Don't manage borrowed pointer */ }) {}
};

template <bool is_const>
struct point_ref : data_ref {
protected:
	using data_t = shared_uint<is_const>;
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

	typename shared_data<is_const>::type* operator&() {
		return data.get();
	}
};

} // namespace detail
} // namespace sycl
} // namespace cl
