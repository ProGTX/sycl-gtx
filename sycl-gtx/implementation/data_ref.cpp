#include "specification\ranges\id.h"
#include "data_ref.h"
#include "gen_source.h"

using namespace cl::sycl::detail;

data_ref& data_ref::operator=(int n) {
	kernel_::source::add(name + " = " + std::to_string(n));
	return *this;
}

data_ref& data_ref::operator=(::cl::sycl::id<1> index) {
	DSELF() << "not implemented";
	if(assignable) {
		kernel_::source::add(name + " = " + index[0].name);
	}
	else {
		// TODO: Error
	}
	return *this;
}

data_ref& data_ref::operator=(data_ref dref) {
	if(assignable) {
		kernel_::source::add(name + " = " + dref.name);
	}
	else {
		// TODO: Error
	}
	return *this;
}

data_ref& data_ref::operator+(data_ref dref) {
	assignable = false;
	name += string_class(" + ") + dref.name;
	return *this;
}
