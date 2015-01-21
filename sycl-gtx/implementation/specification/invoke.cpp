#include "invoke.h"

using namespace cl::sycl;

using namespace detail::kernel_;

source* source::scope = nullptr;

void source::execute() {
	// TODO: Create kernel source
}

string_class source::get_source() {
	// TODO: Create kernel source
	return "";
}
