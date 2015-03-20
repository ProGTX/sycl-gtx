#include "gen_source.h"

#include "specification\accessor.h"
#include "specification\kernel.h"
#include "specification\program.h"
#include "specification\queue.h"

using namespace cl::sycl;
using namespace detail::kernel_;

source* source::scope = nullptr;

// Creates kernel source
string_class source::get_code() {
	string_class src;
	static const char newline = '\n';

	src = src + "__kernel void " + kernelName + "(" + generate_accessor_list() + ") {" + newline;

	for(auto& line : lines) {
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
		list += acc.second.type_name + " ";
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

void source::compile_command(queue* q, source src, detail::shared_unique<kernel> kern) {
	program p(src.get_code(), q);

	cl_int clError;
	cl_kernel k = clCreateKernel(p.get(), src.kernelName.c_str(), &clError);
	// TODO: Handle error

	int i = 0;
	for(auto& acc : src.resources) {
		clError = clSetKernelArg(k, i, sizeof(cl_mem), acc.second.argument);
		// TODO: Handle error
		++i;
	}

	*kern = std::unique_ptr<kernel>(new kernel(k));
}

detail::shared_unique<kernel> source::compile() {
	auto kern = detail::shared_unique<kernel>(new std::unique_ptr<kernel>());
	cmd_group::add(compile_command, *this, kern);
	return kern;
}
