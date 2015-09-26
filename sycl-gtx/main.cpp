#include "tests\tests.h"

#include <debug.h>
#include <sycl.hpp>
#include <string>
#include <map>
#include <Windows.h>

#define TRY_CATCH_ERRORS 1

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
#if TRY_CATCH_ERRORS
		try {
#endif
			result = test.second();
#if TRY_CATCH_ERRORS
		}
		catch(cl::sycl::exception& e) {
			debug() << "cl::sycl::exception while testing" << test.first << e.what();
		}
		catch(std::exception& e) {
			debug() << "std::exception while testing" << test.first << e.what();
		}
#endif
		debug();
		debug() << test.first << (result ? "successful" : "failed");
	}

	system("pause");
	return 0;
}
