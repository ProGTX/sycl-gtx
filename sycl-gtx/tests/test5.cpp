#include "tests.h"

#include "../implementation/specification/refc.h"

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

	for(int i = 0; i < num_platforms; ++i) {
		debug();
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
	}

	if(success && num_platforms > 0) {
		debug();
#define USE_SYCL 1
#if USE_SYCL
		using namespace cl::sycl;

		auto platform = refc::allocate<cl_platform_id>();
		auto platform_id = platform.get();
		cl_uint num_devices;
		auto device = refc::allocate<cl_device_id>();
		auto device_id = device.get();
		cl_int ret;

		debug() << "getting platform";
		check_error(clGetPlatformIDs(1, &platform_id, &num_platforms));
		debug() << "getting device";
		check_error(clGetDeviceIDs(platform_id, CL_DEVICE_TYPE_DEFAULT, 1, &device_id, &num_devices));
		debug() << "creating context";
		auto context = refc::allocate(clCreateContext(NULL, 1, &device_id, NULL, NULL, &ret), clReleaseContext);
		check_error(ret);
#else
		cl_context context = NULL;
		cl_device_id device_id = NULL;
		cl_platform_id platform_id = NULL;
		cl_uint num_devices;
		cl_int ret;

		debug() << "getting platform";
		check_error(clGetPlatformIDs(1, &platform_id, &num_platforms));
		debug() << "getting device";
		check_error(clGetDeviceIDs(platform_id, CL_DEVICE_TYPE_DEFAULT, 1, &device_id, &num_devices));
		debug() << "creating context";
		context = clCreateContext(NULL, 1, &device_id, NULL, NULL, &ret);

		debug() << "releasing context";
		check_error(clReleaseContext(context));
		debug() << "releasing device";
		check_error(clReleaseDevice(device_id));
#endif
	}

	return success;
}
