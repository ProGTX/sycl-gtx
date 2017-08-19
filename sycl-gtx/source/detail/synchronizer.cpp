#include "SYCL/detail/synchronizer.h"

#include "SYCL/accessor.h"
#include "SYCL/buffer_base.h"
#include "SYCL/queue.h"

using namespace cl::sycl;
using namespace detail;

std::set<queue*> synchronizer::queues;
std::map<accessor_base*, buffer_base*> synchronizer::host_accessors;

void synchronizer::wait_on_queues(buffer_base* buf) {
  for (auto&& q : queues) {
    if (q->buffers_in_use.count(buf) > 0) {
      q->wait();
    }
  }
}

void synchronizer::flush_queues(buffer_base* buf) {
  for (auto&& q : queues) {
    if (q->buffers_in_use.count(buf) > 0) {
      q->flush();
    }
  }
}

void synchronizer::add(queue* q) {
  queues.insert(q);
}

void synchronizer::remove(queue* q) {
  queues.erase(q);
}

void synchronizer::add(accessor_base* acc, buffer_base* buf) {
  DSELF() << acc << buf;
  host_accessors.emplace(acc, buf);
  wait_on_queues(buf);
}

void synchronizer::remove(accessor_base* acc, buffer_base* buf) {
  host_accessors.erase(acc);
  flush_queues(buf);
}

bool synchronizer::can_flush(
    const std::set<detail::buffer_base*>& buffers_in_use) {
  {
    auto d = DSELF();
    d << "buffers_in_use";
    for (auto& buf : buffers_in_use) {
      d << buf;
    }
  }
  {
    auto d = DSELF();
    d << "host_accessors";
    for (auto&& acc : host_accessors) {
      d << "{" << acc.first << acc.second << "}";
    }
  }
  for (auto&& acc : host_accessors) {
    if (buffers_in_use.count(acc.second) > 0) {
      return false;
    }
  }
  return true;
}
