#include "SYCL/handler.h"

#include "SYCL/context.h"
#include "SYCL/queue.h"

using namespace cl::sycl;
using namespace detail;

context handler::get_context(queue* q) {
  return q->get_context();
}
