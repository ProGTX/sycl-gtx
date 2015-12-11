#include "handler.h"

#include "context.h"
#include "queue.h"

using namespace cl::sycl;
using namespace detail;

context handler::get_context(queue* q) {
	return q->get_context();
}
