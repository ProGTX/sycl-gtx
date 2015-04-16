#include "program.h"

#include "queue.h"
#include "../debug.h"

using namespace cl::sycl;

program::program(string_class source, queue* q) {
	const char* src = source.c_str();
	size_t length = source.size();
	cl_int error_code;

	auto Context = q->get_context().get();
	auto Device = q->get_device().get();

	cl_program p = clCreateProgramWithSource(Context, 1, &src, &length, &error_code);
	handler.report(error_code);

	prog = refc::allocate<cl_program>(p, clReleaseProgram);

	error_code = clBuildProgram(p, 1, &Device, nullptr, nullptr, nullptr);
	try {
		handler.report(error_code);
	}
	catch(::cl::sycl::exception& e) {
		// http://stackoverflow.com/a/9467325/793006

		// Determine the size of the log
		size_t log_size;
		clGetProgramBuildInfo(p, Device, CL_PROGRAM_BUILD_LOG, 0, nullptr, &log_size);

		// Allocate memory for the log
		auto log = new char[log_size];

		// Get the log
		clGetProgramBuildInfo(p, Device, CL_PROGRAM_BUILD_LOG, log_size, log, nullptr);

		debug() << "Error while compiling program:\n" << log;

		delete[] log;

		throw e;
	}
}
