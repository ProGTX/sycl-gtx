#pragma once

// 2.5.9 Managing object lifetimes
// "All OpenCL objects encapsulated in SYCL objects will be reference-counted and destroyed once all references have been released."

#include <CL/cl.h>
#include <memory>
#include <functional>

namespace cl {
namespace sycl {

namespace refc {

template<class CL_Type>
using releaser = cl_int(CL_API_CALL*)(CL_Type);

template<class CL_Type>
using ptr = std::shared_ptr<typename std::remove_pointer<CL_Type>::type>;

template<class CL_Type>
cl_int CL_API_CALL empty_releaser(CL_Type) {
	return CL_SUCCESS;
}

template<class CL_Type>
static ptr<CL_Type> allocate(releaser<CL_Type> release = &empty_releaser<CL_Type>) {
	return{ nullptr, release };
}

template<class CL_Type>
static ptr<CL_Type> allocate(CL_Type data, releaser<CL_Type> release = &empty_releaser<CL_Type>) {
	return{ data, release };
}

template<class CL_Type>
static ptr<CL_Type> allocate(ptr<CL_Type> pointer, releaser<CL_Type> release = &empty_releaser<CL_Type>) {
	return{ pointer.get(), release };
}

} // namespace refc

} // namespace sycl
} // namespace cl
