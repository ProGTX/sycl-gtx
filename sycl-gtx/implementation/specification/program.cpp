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
	PrintBuildLog(prog, Device);
	if(clError != CL_SUCCESS) {
		debug() << "Failed to build CL program.";
		clReleaseProgram(prog);
	}
}

void program::PrintBuildLog(cl_program Program, cl_device_id Device) {
	cl_build_status buildStatus;
	clGetProgramBuildInfo(Program, Device, CL_PROGRAM_BUILD_STATUS, sizeof(cl_build_status), &buildStatus, NULL);

	// let's print out possible warnings even if the kernel compiled..
	//if(buildStatus == CL_SUCCESS)
	//	return;

	//there were some errors.
	size_t logSize;
	clGetProgramBuildInfo(Program, Device, CL_PROGRAM_BUILD_LOG, 0, NULL, &logSize);
	string_class buildLog(logSize, ' ');

	clGetProgramBuildInfo(Program, Device, CL_PROGRAM_BUILD_LOG, logSize, &buildLog[0], NULL);
	buildLog[logSize] = '\0';

	if(buildStatus != CL_SUCCESS) {
		debug() << "There were build errors!";
	}
	debug() << "Build log:";
	debug() << buildLog;
}
