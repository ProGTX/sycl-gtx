#pragma once

// 2.5.8 Managing object lifetimes
// "All OpenCL objects encapsulated in SYCL objects will be reference-counted and destroyed once all references have been released."

#include <CL/cl.h>
#include <memory>

namespace cl {
namespace sycl {

namespace detail {

template <class CL_Type>
using cl_resource_f = cl_int(CL_API_CALL*)(CL_Type);

template<class CL_Type>
cl_int CL_API_CALL cl_do_nothing(CL_Type) {
	return CL_SUCCESS;
}

template<class CL_Type>
using refc_ptr = std::shared_ptr<typename std::remove_pointer<CL_Type>::type>;

template <class CL_Type, cl_resource_f<CL_Type> retain = cl_do_nothing<CL_Type>, cl_resource_f<CL_Type> release = &cl_do_nothing<CL_Type>>
class refc : public refc_ptr<CL_Type> {
private:
	using Base = refc_ptr<CL_Type>;

public:
	refc()
		: Base(nullptr, release) {}
	
	refc(CL_Type data)
		: Base(data, release) {}

	refc(const refc& copy)
		: Base((const Base&)copy) {
		retain(get());
	}

	refc(refc&& move)
		: Base(std::move(move)) {}
};

} // namespace detail

} // namespace sycl
} // namespace cl
