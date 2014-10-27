#pragma once

// 3.2.4 Device selection class

#include "device.h"

namespace cl {
namespace sycl {

// The class device_selector is an abstract class which enables the SYCL runtime to choose the best device based
// on heuristics specified by the user, or by one of the built-in device selectors

class device_selector {
public:
	virtual int operator()(device dev) = 0;
};

// Built-in device selectors:
// Class name: gpu_selector
// Class name: cpu_selector
// Class name: host_selector

} // namespace sycl
} // namespace cl
