#include "SYCL/detail/kernel_name.h"

using namespace cl::sycl;
using namespace detail;

::size_t kernel_name::current_count = 0;
