#include "specification\ranges\id.h"
#include "data_ref.h"
#include "gen_source.h"

using namespace cl::sycl::detail;

const ::cl::sycl::string_class data_ref::open_parenthesis = "(";

data_ref& data_ref::operator=(int n) {
	kernel_::source::add(name + " = " + std::to_string(n));
	return *this;
}

data_ref& data_ref::operator=(::cl::sycl::id<1> index) {
	kernel_::source::add(name + " = " + index[0].name);
	return *this;
}

data_ref& data_ref::operator=(data_ref dref) {
	kernel_::source::add(name + " = " + dref.name);
	return *this;
}
