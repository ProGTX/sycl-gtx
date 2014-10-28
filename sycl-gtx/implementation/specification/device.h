#pragma once

// Device classes

#include "../debug.h"

namespace cl {
namespace sycl {

// 3.2.2 Device class
// Encapsulates a cl_device_id and a cl_platform_id
class device {
public:
	device() {
		DSELF() << "not implemented";
	}
};


// 3.2.4 Device selection class
// The class device_selector is an abstract class which enables the SYCL runtime to choose the best device based
// on heuristics specified by the user, or by one of the built-in device selectors
class device_selector {
public:
	device_selector() {
		DSELF() << "not implemented";
	}
	virtual int operator()(device dev) = 0;
};

// Built-in device selectors:
// Class name: gpu_selector
// Class name: cpu_selector
// Class name: host_selector

} // namespace sycl
} // namespace cl
