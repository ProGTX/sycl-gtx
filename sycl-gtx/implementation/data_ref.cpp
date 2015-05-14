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
	return (index.type == id_ref::type::global ? id_global_name : id_local_name) + "0";
}

// TODO: local id
string_class data_ref::get_name(id<2> index) {
	return detail::id_global_all_name;
}
string_class data_ref::get_name(id<3> index) {
	return detail::id_global_all_name;
}

id_ref::id_ref(int n, size_t* value, type access_type)
	:	data_ref(
			(access_type == type::global ? id_global_name : id_local_name) + std::to_string(n)
		),
		value(value) {}
