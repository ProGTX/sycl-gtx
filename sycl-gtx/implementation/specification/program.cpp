#include "program.h"

#include "queue.h"
#include "../debug.h"

using namespace cl::sycl;

program::program(cl_program clProgram, const context& context, vector_class<device> deviceList)
	: prog(clProgram), ctx(context), devices(deviceList) {}

program::program(const context& context)
	: program(context, context.get_devices()) {}

program::program(const context& context, vector_class<device> deviceList)
	: program(nullptr, context, deviceList) {}

program::program(const context& context, cl_program clProgram)
	: program(clProgram, context, context.get_devices()) {}


void program::compile(string_class compile_options) {
	auto& src = kernel_sources.back();
	auto code = src.get_code();

	debug() << "Compiled kernel:";
	debug() << code;

	const char* code_p = code.c_str();
	size_t length = code.size();
	cl_int error_code;

	prog = clCreateProgramWithSource(ctx.get(), 1, &code_p, &length, &error_code);
	detail::error::report(error_code);

	auto device_pointers = detail::get_cl_array(devices);

	error_code = clCompileProgram(
		prog.get(), devices.size(), device_pointers.data(), compile_options.c_str(),
		0, nullptr, nullptr, nullptr, nullptr
	);

	try {
		detail::error::report(error_code);
	}
	catch(::cl::sycl::exception& e) {
		for(auto& d : devices) {
			report_compile_error(d);
		}
		throw e;
	}
}

void program::report_compile_error(device& dev) {
	// http://stackoverflow.com/a/9467325/793006

	// Determine the size of the log
	size_t log_size;
	clGetProgramBuildInfo(prog.get(), dev.get(), CL_PROGRAM_BUILD_LOG, 0, nullptr, &log_size);

	// Allocate memory for the log
	auto log = new char[log_size];

	// Get the log
	clGetProgramBuildInfo(prog.get(), dev.get(), CL_PROGRAM_BUILD_LOG, log_size, log, nullptr);

	debug() << "Error while compiling program for device" << dev.get_info<info::device::name>() << "\n" << log;

	delete[] log;
}

void program::link(string_class linking_options) {
	if(linked) {
		// TODO: Error?
	}

	auto device_pointers = detail::get_cl_array(devices);
	auto p = prog.get();
	cl_int error_code;

	prog = clLinkProgram(
		ctx.get(),
		device_pointers.size(),
		device_pointers.data(),
		linking_options.c_str(),
		1,
		&p,
		nullptr,
		nullptr,
		&error_code
	);
	detail::error::report(error_code);

	auto& src = kernel_sources.back();
	src.create_kernel(*this);

	linked = true;
}
