#include "gen_source.h"

#include "specification\accessor.h"
#include "specification\buffer.h"
#include "specification\error_handler.h"
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

string_class source::generate_accessor_list() const {
	string_class list;
	if(resources.empty()) {
		return list;
	}

	for(auto& acc : resources) {
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
	error::report(q, clError);

	int i = 0;
	for(auto& acc : src.resources) {
		auto mem = acc.second.buffer->device_data.get();
		clError = clSetKernelArg(k, i, sizeof(cl_mem), &mem);
		error::report(q, clError);
		++i;
	}

	*kern = std::unique_ptr<kernel>(new kernel(k));
}

// Note: MSVC2013 editor reports errors on cmd_group::add, but the code compiles and links

detail::shared_unique<kernel> source::compile() const {
	auto kern = detail::shared_unique<kernel>(new std::unique_ptr<kernel>());
	cmd_group::add(compile_command, *this, kern);
	return kern;
}

void source::enqueue_write_buffers() const {
	for(auto& acc : resources) {
		if(	acc.second.mode == access::write		||
			acc.second.mode == access::read_write	||
			acc.second.mode == access::discard_read_write
		) {
			cmd_group::add(buffer_base::enqueue_write_command, acc.second.buffer);
		}
	}
}


void source::enqueue_task_command(queue* q, detail::shared_unique<kernel> kern) {
	(*kern)->enqueue_task(q);
}

void source::enqueue_task(detail::shared_unique<kernel> kern) {
	cmd_group::add(enqueue_task_command, kern);
}

void source::enqueue_read_buffers() const {
	for(auto& acc : resources) {
		if(	acc.second.mode == access::read			||
			acc.second.mode == access::read_write	||
			acc.second.mode == access::discard_read_write
		) {
			cmd_group::add(buffer_base::enqueue_read_command, acc.second.buffer);
		}
	}
}
