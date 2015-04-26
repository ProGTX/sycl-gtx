#include "specification\ranges\id.h"
#include "data_ref.h"
#include "gen_source.h"

using namespace cl::sycl;
using detail::data_ref;
using detail::id_ref;

const string_class data_ref::open_parenthesis = "(";

const char data_ref::assign::normal[]	= " = ";
const char data_ref::assign::add[]		= " += ";
const char data_ref::assign::subtract[]	= " -= ";
const char data_ref::assign::multiply[]	= " *= ";
const char data_ref::assign::divide[]	= " /= ";
const char data_ref::assign::modulo[]	= " %= ";

void data_ref::kernel_add(string_class line) {
	kernel_::source::add(line);
}

string_class data_ref::get_name(id<1> index) {
	return detail::id_base_name + "0";
}

id_ref::id_ref(int n, size_t* value)
	: data_ref(id_base_name + std::to_string(n)), value(value) {}
