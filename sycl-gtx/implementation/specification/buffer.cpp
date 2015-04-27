#include "buffer.h"

using namespace cl::sycl;
using detail::buffer_base;

unsigned int buffer_base::counter = 0;

void buffer_base::generate_name() {
	resource_name = string_class("_sycl_buf_") + std::to_string(++counter);
}
