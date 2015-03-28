#include "data_ref.h"
#include "gen_source.h"

using namespace cl::sycl::detail;

const data_ref& data_ref::operator=(int n) const {
	DSELF() << "not implemented";
	kernel_::source::add(name + " = " + std::to_string(n));
	return *this;
}
