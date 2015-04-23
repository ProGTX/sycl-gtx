#include "specification\ranges\id.h"
#include "data_ref.h"
#include "gen_source.h"

using namespace cl::sycl;
using detail::data_ref;

const string_class data_ref::open_parenthesis = "(";

template
data_ref& data_ref::operator=(id<1> n);
template
data_ref& data_ref::operator=(int n);

template <class T, data_ref::is_compatible_t<T>*>
void data_ref::assign(T n) {
	kernel_::source::add(name + " = " + get_name(n));
}

data_ref& data_ref::operator=(data_ref dref) {
	assign(dref);
	return *this;
}

template <class T, data_ref::is_compatible_t<T>*>
data_ref& data_ref::operator=(T n) {
	assign(n);
	return *this;
}

string_class data_ref::get_name(id<1> index) {
	return detail::id_base_name + "0";
}
