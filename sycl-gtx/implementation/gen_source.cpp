#include "gen_source.h"
#include "specification\accessor.h"

using namespace cl::sycl;
using namespace detail::kernel_;

source* source::scope = nullptr;

// Creates kernel source
string_class source::get_code() {
	string_class src;
	static const char newline = '\n';

	src = src + "__kernel void " + kernelName + "(" + generate_accessor_list() + ") {" + newline;

	for(auto&& line : lines) {
		src += std::move(line) + newline;
	}

	src = src + "}" + newline;

	return src;
}

string_class source::generate_accessor_list() {
	string_class list;
	if(resources.empty()) {
		return list;
	}

	for(auto&& acc : resources) {
		list += get_name(acc.second.target) + " ";
		if(acc.second.mode == access::mode::read) {
			list += "const ";
		}
		list += acc.second.type + " ";
		list += acc.first + ", ";
	}

	// 2 to get rid of the last comma and space
	return list.substr(0, list.length() - 2);
}

string_class source::get_name(access::target target) {
	// TODO: All cases
	switch(target) {
		case access::target::global_buffer:
			return "__global";
		case access::target::constant_buffer:
			return "__constant";
		case access::target::local:
			return "__local";
		default:
			return "";
	}
}
