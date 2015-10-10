#pragma once

#include <type_traits>

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

	ptr_or_val& operator=(T n) {
		if(is_owner) {
			data = reinterpret_cast<T*>(n);
		}
		else {
			*data = n;
		}
		return *this;
	}

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

	ptr_or_val<T*> operator&() {
		return ptr_or_val<T*>(&data);
	}
	ptr_or_val<typename std::remove_pointer<T>::type> operator*() {
		return ptr_or_val<typename std::remove_pointer<T>::type>(*data);
	}
};

} // namespace detail
} // namespace sycl
} // namespace cl
