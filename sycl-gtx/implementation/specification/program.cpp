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
	handler.report(error_code);
}
