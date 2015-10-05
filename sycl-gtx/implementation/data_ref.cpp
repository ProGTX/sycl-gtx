#include "specification\ranges\id.h"
#include "data_ref.h"
#include "src_handlers\kernel_source.h"

using namespace cl::sycl;
using namespace detail;

void detail::kernel_add(string_class line) {
	kernel_::source::add(line);
}

const string_class data_ref::open_parenthesis = "(";
