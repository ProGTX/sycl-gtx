#include "synchronizer.h"

#include "specification/buffer_base.h"
#include "specification/queue.h"

using namespace cl::sycl;
using namespace detail;

std::set<queue*> synchronizer::queues;

void synchronizer::add(queue* q) {
	queues.insert(q);
}

void synchronizer::remove(queue* q) {
	queues.erase(q);
}

void synchronizer::barrier(buffer_base* buf) {
	for(auto&& q : queues) {
		if(q->buffers_in_use.count(buf) > 0) {
			q->wait();
		}
	}
}
