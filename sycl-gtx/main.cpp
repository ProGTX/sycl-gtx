#include "tests\tests.h"

#include <debug.h>
#include <sycl.hpp>
#include <string>
#include <map>
#include <Windows.h>

int main() {
	debug() << "SYCL Provisional Specification.";
	
	cl::sycl::device d;
	debug() << "Default device information:";
	debug() << d.get_info<CL_DEVICE_NAME>();
	debug() << d.get_info<CL_DEVICE_OPENCL_C_VERSION>();
	debug() << d.get_info<CL_DEVICE_PROFILE>();
	debug() << d.get_info<CL_DEVICE_VERSION>();
	debug() << d.get_info<CL_DRIVER_VERSION>();

	std::map<std::string, bool(*)()> tests{
		//{ "test1", test1 },
		//{ "test2", test2 },
		//{ "test3", test3 },
		//{ "test4", test4 },
		//{ "test5", test5 },
		//{ "test6", test6 },
		//{ "test7", test7 },
		//{ "test8", test8 },
	};

	for(auto&& test : tests) {
		debug();
		debug() << "starting" << test.first;
		debug();
		auto result(test.second());
		debug();
		debug() << test.first << (result ? "successful" : "failed");
	}

	system("pause");
	return 0;
}
