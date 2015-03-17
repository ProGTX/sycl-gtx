#include "program.h"

#include "queue.h"
#include "../debug.h"

using namespace cl::sycl;

// TODO: Very rough
program::program(string_class source, queue* q) {
	const char* src = source.c_str();
	size_t length = source.size();
	cl_int clError;

	auto Context = q->get_context().get();
	auto Device = q->get_device().get();

	cl_program prog = clCreateProgramWithSource(Context, 1, &src, &length, &clError);
	if(clError != CL_SUCCESS) {
		debug() << "Failed to create CL program from source.";
		return;
	}

	clError = clBuildProgram(prog, 1, &Device, nullptr, nullptr, nullptr);
	//PrintBuildLog(prog, Device);
	if(clError != CL_SUCCESS) {
		debug() << "Failed to build CL program.";
		clReleaseProgram(prog);
	}

}
