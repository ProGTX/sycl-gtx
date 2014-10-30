#include "tests.h"

bool test5() {
	static const int MAX_PLATFORMS = 10;
	static const int BUFFER_SIZE = 1024;
	cl_platform_id platforms[MAX_PLATFORMS];
	cl_uint num_platforms = 0;
	char buffer[BUFFER_SIZE];

	bool success(true);
	auto check_error = [&success](cl_int ret_val) {
		if(ret_val != CL_SUCCESS) {
			debug() << "OpenCL error:" << ret_val;
			success = false;
		}
	};

	check_error(clGetPlatformIDs(MAX_PLATFORMS, platforms, &num_platforms));

	debug() << "num_platforms:" << num_platforms;
	debug();

	for(int i = 0; i < num_platforms; ++i) {
		debug() << "platform num:\t" << i;
		check_error(clGetPlatformInfo(platforms[i], CL_PLATFORM_PROFILE, BUFFER_SIZE, buffer, NULL));
		debug() << "profile:\t" << buffer;
		check_error(clGetPlatformInfo(platforms[i], CL_PLATFORM_VERSION, BUFFER_SIZE, buffer, NULL));
		debug() << "version:\t" << buffer;
		check_error(clGetPlatformInfo(platforms[i], CL_PLATFORM_NAME, BUFFER_SIZE, buffer, NULL));
		debug() << "name:\t\t" << buffer;
		check_error(clGetPlatformInfo(platforms[i], CL_PLATFORM_VENDOR, BUFFER_SIZE, buffer, NULL));
		debug() << "vendor:\t\t" << buffer;
		check_error(clGetPlatformInfo(platforms[i], CL_PLATFORM_EXTENSIONS, BUFFER_SIZE, buffer, NULL));
		debug() << "extensions:\t" << buffer;
		debug();
	}

	return success;
}
