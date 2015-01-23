#include "invoke.h"

#include "accessor.h"

using namespace cl::sycl;

using namespace detail::kernel_;

source* source::scope = nullptr;

// Creates kernel source
string_class source::get() {
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
		list += get_name(std::get<2>(acc.second)) + " ";
		if(std::get<1>(acc.second) == access::mode::read) {
			list += "const ";
		}
		list += std::get<0>(acc.second) + " ";
		list += acc.first->resource_name() + ", ";
	}

	// 2 to get rid of the last comma and space
	return list.substr(0, list.length() - 2);
}

string_class source::get_name(access::target target) {
	// TODO
	return "";
}
