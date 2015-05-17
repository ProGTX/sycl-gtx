#pragma once

// 3.8 Data Types

#include "../common.h"
#include "../counter.h"
#include "../data_ref.h"

namespace cl {
namespace sycl {

// 3.8.1 Vector types

namespace detail {

template <typename dataT, int numElements>
class vec_base : public detail::counter<vec_base<dataT, numElements>>, public detail::data_ref {
private:
	static string_class type_name() {
		return detail::type_string<dataT>() + (numElements == 1 ? "" : std::to_string(numElements));
	}

	string_class generate_name() const {
		return '_' + type_name() + '_' + std::to_string(counter_id);
	}

public:
	vec_base()
		: counter(), data_ref(generate_name()) {
		detail::kernel_add(type_name() + ' ' + name);
	}

	template <class T>
	vec_base(T n)
		: counter(), data_ref(generate_name()) {
		detail::kernel_add(type_name() + ' ' + name + " = " + get_name(n));
	}
};

} // namespace detail


#define SYCL_VEC_INHERIT(type)						\
	type()											\
		: Base() {}									\
	template <class T>								\
	type(T n)										\
		: Base(n) {}								\
	template <class T>								\
	data_ref& operator=(T n) {						\
		return data_ref::operator=(n);				\
	}												\
	template <class T>								\
	friend data_ref operator*(T n, type&& elem) {	\
		return n * (data_ref&&)elem;				\
	}

template <typename dataT, int numElements>
class vec : public detail::vec_base<dataT, numElements> {
private:
	using Base = detail::vec_base<dataT, numElements>;
public:
	SYCL_VEC_INHERIT(vec)
};


// 3.9.1 Description of the built-in types available for SYCL host and device

#define SYCL_VEC_SIGNED(type, numElements)						\
	class type##numElements : public vec<type, numElements> {	\
		using Base = vec<type, numElements>;					\
	public:														\
		SYCL_VEC_INHERIT(type##numElements)						\
	};

#define SYCL_VEC_UNSIGNED(type, numElements)								\
	class u##type##numElements : public vec<unsigned type, numElements> {	\
		using Base = vec<unsigned type, numElements>;						\
	public:																	\
		SYCL_VEC_INHERIT(u##type##numElements)								\
	};

#define SYCL_ADD_SIGNED_vec(type)	\
	SYCL_VEC_SIGNED(type, 1)		\
	SYCL_VEC_SIGNED(type, 2)		\
	SYCL_VEC_SIGNED(type, 3)		\
	SYCL_VEC_SIGNED(type, 4)		\
	SYCL_VEC_SIGNED(type, 8)		\
	SYCL_VEC_SIGNED(type, 16)

#define SYCL_ADD_UNSIGNED_vec(type)	\
	SYCL_VEC_UNSIGNED(type, 1)		\
	SYCL_VEC_UNSIGNED(type, 2)		\
	SYCL_VEC_UNSIGNED(type, 3)		\
	SYCL_VEC_UNSIGNED(type, 4)		\
	SYCL_VEC_UNSIGNED(type, 8)		\
	SYCL_VEC_UNSIGNED(type, 16)

SYCL_ADD_SIGNED_vec(bool)
SYCL_ADD_SIGNED_vec(int)
SYCL_ADD_SIGNED_vec(char)
SYCL_ADD_SIGNED_vec(short)
SYCL_ADD_SIGNED_vec(long)
SYCL_ADD_SIGNED_vec(float)
SYCL_ADD_SIGNED_vec(double)

SYCL_ADD_UNSIGNED_vec(int)
SYCL_ADD_UNSIGNED_vec(char)
SYCL_ADD_UNSIGNED_vec(short)
SYCL_ADD_UNSIGNED_vec(long)

#undef SYCL_ADD_SIGNED_vec
#undef SYCL_ADD_UNSIGNED_vec
#undef SYCL_VEC_INHERIT
#undef SYCL_VEC_SIGNED
#undef SYCL_VEC_UNSIGNED

} // namespace sycl
} // namespace cl
