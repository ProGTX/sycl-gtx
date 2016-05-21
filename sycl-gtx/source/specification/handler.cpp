#include "specification/handler.h"

#include "specification/context.h"
#include "specification/queue.h"

using namespace cl::sycl;
using namespace detail;

context handler::get_context(queue* q) {
	return q->get_context();
}
