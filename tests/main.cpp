#include "tests.h"

#include <string>
#include <map>

#define TRY_CATCH_ERRORS 1

int main() {
	using namespace std;

	debug() << "SYCL 1.2 Final Specification, Provisional Implementation.";
	
	std::map<std::string, bool(*)()> tests{
		{ "test1", test1 },
		{ "test2", test2 },
		{ "test3", test3 },
		{ "test4", test4 },
		{ "test5", test5 },
		{ "test6", test6 },
		{ "test7", test7 },
		{ "test8", test8 },
		{ "test9", test9 },
		{ "test10", test10 },
		{ "test11", test11 },
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

	cout << "Press any key to exit ..." << endl;
	cin.get();

	return 0;
}
