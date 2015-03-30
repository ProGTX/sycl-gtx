#include "data_ref.h"
#include "gen_source.h"

using namespace cl::sycl::detail;

data_ref& data_ref::operator=(int n) {
	DSELF() << "not implemented";
	kernel_::source::add(name + " = " + std::to_string(n));
	return *this;
}

data_ref& data_ref::operator=(data_ref dref) {
	DSELF() << "not implemented";
	if(assignable) {
		// TODO: Somehow causes a segfault
		kernel_::source::add(name + " = " + dref.name);
	}
	else {
		// TODO: Error
	}
	return *this;
}

data_ref& data_ref::operator+(data_ref dref) {
	DSELF() << "not implemented";
	assignable = false;
	name += string_class(" + ") + dref.name;
	return *this;
}
