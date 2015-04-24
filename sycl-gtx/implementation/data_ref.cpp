#include "specification\ranges\id.h"
#include "data_ref.h"
#include "gen_source.h"

using namespace cl::sycl;
using detail::data_ref;

const string_class data_ref::open_parenthesis = "(";

const char data_ref::assign::normal[]	= " = ";

void data_ref::kernel_add(string_class line) {
	kernel_::source::add(line);
}

string_class data_ref::get_name(id<1> index) {
	return detail::id_base_name + "0";
}
