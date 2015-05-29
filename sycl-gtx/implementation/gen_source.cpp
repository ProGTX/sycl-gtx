#include "gen_source.h"

#include "specification\accessor\buffer.h"
#include "specification\buffer.h"
#include "specification\command_group.h"
#include "specification\error_handler.h"
#include "specification\kernel.h"
#include "specification\program.h"
#include "specification\queue.h"

using namespace cl::sycl;
using namespace detail::kernel_;

const string_class source::resource_name_root = "_sycl_buf";
int source::num_resources = 0;
int source::num_kernels = 0;
source* source::scope = nullptr;

bool source::in_scope() {
	return scope != nullptr;
}

void source::enter(source& src) {
	scope = &src;
	num_resources = 0;
}

source source::exit(source& src) {
	scope = nullptr;
	return src;
}

// Creates kernel source
string_class source::get_code() {
	if(!final_code.empty()) {
		return final_code;
	}

	static const char newline = '\n';

	final_code = string_class("__kernel void ") + kernel_name + "(" + generate_accessor_list() + ") {" + newline;

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
		list += get_name(acc.second.acc.target) + " ";
		if(acc.second.acc.mode == access::mode::read) {
			list += "const ";
		}
		list += acc.second.type_name + " ";
		list += acc.second.resource_name + ", ";
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
	cl_kernel k = clCreateKernel(p.get(), src.kernel_name.c_str(), &error_code);
	detail::error::report(error_code);

	int i = 0;
	for(auto& acc : src.resources) {
		if(acc.second.acc.target == access::local) {
			error_code = clSetKernelArg(k, i, acc.second.size, nullptr);
			detail::error::report(error_code);
		}
		else {
			auto mem = acc.second.acc.data->device_data.get();
			error_code = clSetKernelArg(k, i, acc.second.size, &mem);
			detail::error::report(error_code);
		}
		++i;
	}

	*kern = unique_ptr_class<kernel>(new kernel(k));
}

// Note: MSVC2013 editor reports errors on command::group_::add, but the code compiles and links

detail::shared_unique<kernel> source::compile() const {
	auto kern = detail::shared_unique<kernel>(new unique_ptr_class<kernel>());
	command::group_::add(compile_command, __func__, *this, kern);
	return kern;
}

void source::write_buffers_to_device() const {
	for(auto& acc : resources) {
		auto mode = acc.second.acc.mode;
		if(
			mode == access::write				||
			mode == access::discard_write		||
			mode == access::discard_read_write	||
			acc.second.acc.target == access::local
		) {
			// Don't need to copy data that won't be used
			continue;
		}
		command::group_::add(
			acc.second.acc,
			access::write,
			buffer_base::enqueue_command,
			__func__,
			acc.second.acc.data,
			&clEnqueueWriteBuffer
		);
	}
}

void source::enqueue_task_command(queue* q, detail::shared_unique<kernel> kern) {
	(*kern)->enqueue_task(q);
}

void source::enqueue_task(detail::shared_unique<kernel> kern) const {
	command::group_::add(enqueue_task_command, __func__, kern);
}

void source::read_buffers_from_device() const {
	for(auto& acc : resources) {
		if(
			acc.second.acc.mode == access::read	||
			acc.second.acc.target == access::local
		) {
			// Don't need to read back read-only buffers
			continue;
		}
		command::group_::add(
			acc.second.acc,
			access::read,
			buffer_base::enqueue_command,
			__func__,
			acc.second.acc.data,
			reinterpret_cast<buffer_base::clEnqueueBuffer_f>(&clEnqueueReadBuffer)
		);
	}
}
