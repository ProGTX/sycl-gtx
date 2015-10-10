#pragma once

#include <type_traits>

namespace cl {
namespace sycl {
namespace detail {

template <typename T>
struct ptr_or_val {
private:
	bool is_owner;
	void* data;

	T* get_ptr() const {
		return reinterpret_cast<T*>(data);
	}

public:
	static_assert(sizeof(T) <= sizeof(void*), "Type T is too big to store as ptr_or_val");

	ptr_or_val(nullptr_t, T value)
		: is_owner(true), data(reinterpret_cast<void*>(value)) {}
	ptr_or_val()
		: ptr_or_val(nullptr, 0) {}
	ptr_or_val(T* ptr)
		: is_owner(false), data(ptr) {}

	ptr_or_val& operator=(T n) {
		if(is_owner) {
			data = reinterpret_cast<T*>(n);
		}
		else {
			*get_ptr() = n;
		}
		return *this;
	}

	operator T() const {
		if(is_owner) {
			return reinterpret_cast<T>(data);
		}
		else {
			return *get_ptr();
		}
	}
	operator T&() {
		if(is_owner) {
			return reinterpret_cast<T&>(data);
		}
		else {
			return *get_ptr();
		}
	}

	ptr_or_val<T*> operator&() {
		return ptr_or_val<T*>(reinterpret_cast<T**>(&data));
	}
	ptr_or_val<typename std::remove_pointer<T>::type> operator*() {
		return ptr_or_val<typename std::remove_pointer<T>::type>(reinterpret_cast<T>(*data));
	}
};

} // namespace detail
} // namespace sycl
} // namespace cl
