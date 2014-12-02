#include "tests.h"

#define DISPLAY_TRAIT(object, trait)	\
debug(#trait ":\t") << object.get_info<trait>();

class handler : public cl::sycl::error_handler {
public:
	virtual void report_error(cl::sycl::exception& error) const override {
		debug("OpenCL error:\t") << ((cl::sycl::cl_exception&)error).get_cl_code();
	}
};

bool test5() {
	debug();
	using namespace cl::sycl;

	debug("= Getting all platforms");
	auto platforms = platform::get_platforms();
	debug("platforms size:") << platforms.size();
	int i = 0;
	for(auto& p : platforms) {
		debug();
		debug("== platform") << (i++);
		DISPLAY_TRAIT(p, CL_PLATFORM_PROFILE);
		DISPLAY_TRAIT(p, CL_PLATFORM_VERSION);
		DISPLAY_TRAIT(p, CL_PLATFORM_NAME);
		DISPLAY_TRAIT(p, CL_PLATFORM_VENDOR);
		//DISPLAY_TRAIT(p, CL_PLATFORM_EXTENSIONS);
	}
	debug();
	debug();

	debug("= Getting all devices");
	std::vector<std::vector<device>> devices;
	int j;
	for(i = 0; i < platforms.size(); ++i) {
		debug();
		debug("== platform") << i;
		devices.push_back(platforms[i].get_devices(CL_DEVICE_TYPE_ALL));
		j = 0;
		for(auto& d : devices[i]) {
			debug();
			debug("=== device") << (j++);
			DISPLAY_TRAIT(d, CL_DEVICE_BUILT_IN_KERNELS);
			DISPLAY_TRAIT(d, CL_DEVICE_NAME);
			DISPLAY_TRAIT(d, CL_DEVICE_OPENCL_C_VERSION);
			DISPLAY_TRAIT(d, CL_DEVICE_PROFILE);
			DISPLAY_TRAIT(d, CL_DEVICE_VENDOR);
			DISPLAY_TRAIT(d, CL_DEVICE_VERSION);
			DISPLAY_TRAIT(d, CL_DRIVER_VERSION);
			DISPLAY_TRAIT(d, CL_DEVICE_AVAILABLE);
			DISPLAY_TRAIT(d, CL_DEVICE_COMPILER_AVAILABLE);
			DISPLAY_TRAIT(d, CL_DEVICE_LINKER_AVAILABLE);
			DISPLAY_TRAIT(d, CL_DEVICE_DOUBLE_FP_CONFIG);
			DISPLAY_TRAIT(d, CL_DEVICE_SINGLE_FP_CONFIG);
			DISPLAY_TRAIT(d, CL_DEVICE_PLATFORM);
			DISPLAY_TRAIT(d, CL_DEVICE_TYPE);
			DISPLAY_TRAIT(d, CL_DEVICE_QUEUE_PROPERTIES);
			//DISPLAY_TRAIT(d, CL_DEVICE_EXTENSIONS);
		}
	}
	debug();
	debug();

	debug("= Trying to throw an exception");
	handler h;
	device dd(h);
	devices.clear();
	try {
		devices.push_back(dd.get_devices(CL_DEVICE_TYPE_ALL));
	}
	catch(cl_exception& e) {
		debug("exception:\t") << e.get_description();
	}

	return true;
}
