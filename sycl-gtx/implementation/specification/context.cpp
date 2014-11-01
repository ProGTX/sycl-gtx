#include "context.h"

using namespace cl::sycl;

cl_context context::get() {
	return ctx.get();
}
