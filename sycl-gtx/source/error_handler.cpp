#include "SYCL/error_handler.h"

#include "SYCL/context.h"

using namespace cl::sycl;
using namespace detail;

void error::thrower::report_async(context* thrower, exception_list& list) {
  thrower->asyncHandler(list);
}
