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
	if(!final_code.empty()) {
		return final_code;
	}

	static const char newline = '\n';

	final_code = string_class("__kernel void ") + kernelName + "(" + generate_accessor_list() + ") {" + newline;

	for(auto& line : lines) {
		final_code += std::move(line) + newline;
	}
	lines.clear();

	final_code = final_code + "}" + newline;

	return final_code;
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

	cl_int error_code;
	cl_kernel k = clCreateKernel(p.get(), src.kernelName.c_str(), &error_code);
	error::report(q, error_code);

	int i = 0;
	for(auto& acc : src.resources) {
		auto mem = acc.second.buffer->device_data.get();
		error_code = clSetKernelArg(k, i, sizeof(cl_mem), &mem);
		error::report(q, error_code);
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

void source::write_buffers_to_device() const {
	for(auto& acc : resources) {
		cmd_group::add(
			buffer_base::enqueue_command,
			acc.second.buffer,
			&clEnqueueWriteBuffer
		);
	}
}

void source::enqueue_task_command(queue* q, detail::shared_unique<kernel> kern) {
	(*kern)->enqueue_task(q);
}

void source::enqueue_task(detail::shared_unique<kernel> kern) const {
	cmd_group::add(enqueue_task_command, kern);
}

void source::read_buffers_from_device() const {
	for(auto& acc : resources) {
		cmd_group::add(
			buffer_base::enqueue_command,
			acc.second.buffer,
			reinterpret_cast<buffer_base::clEnqueueBuffer_f>(&clEnqueueReadBuffer)
		);
	}
}
