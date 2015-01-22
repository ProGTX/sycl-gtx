#include "invoke.h"

#include "accessor.h"

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

string_class source::get_name(access::target) {
	// TODO
	return "";
}
