#include "tests\tests.h"

#include <debug.h>
#include <sycl.hpp>
#include <Windows.h>

int main() {
	debug() << "SYCL Provisional Specification.";

	debug() << "starting test1";
	debug("test1") << (test1() ? "successful" : "failed");

	system("pause");
	return 0;
}
