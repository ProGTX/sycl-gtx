#pragma once

#include "SYCL/event.h"

namespace cl {
namespace sycl {

// Forward declaration
class handler;

// TODO(progtx):
class handler_event {
 private:
  friend class handler;

  event kernelEvent;
  event completeEvent;
  event endEvent;

 public:
  event get_kernel() const {
    return kernelEvent;
  }
  event get_complete() const {
    return completeEvent;
  }
  event get_end() const {
    return endEvent;
  }
};

}  // namespace sycl
}  // namespace cl
