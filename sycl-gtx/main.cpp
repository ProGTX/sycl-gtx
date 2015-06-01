#include "tests\tests.h"

#include <debug.h>
#include <sycl.hpp>
#include <string>
#include <map>
#include <Windows.h>

int main() {
	debug() << "SYCL Provisional Specification.";
	
	std::map<std::string, bool(*)()> tests{
		//{ "test1", test1 },
		{ "test2", test2 },
		{ "test3", test3 },
		{ "test4", test4 },
		{ "test5", test5 },
		{ "test6", test6 },
		{ "test7", test7 },
		{ "test8", test8 },
		{ "test9", test9 },
	};

	bool result = false;

	for(auto&& test : tests) {
		debug();
		debug() << "starting" << test.first;
		debug();
		try {
			result = test.second();
		}
		catch(cl::sycl::exception& e) {
			debug() << "cl::sycl::exception while testing" << test.first << e.what();
		}
		catch(std::exception& e) {
			debug() << "std::exception while testing" << test.first << e.what();
		}
		debug();
		debug() << test.first << (result ? "successful" : "failed");
	}

	system("pause");
	return 0;
}
