#pragma once

// 3.2.4 Device selection class

#include <memory>

namespace cl {
namespace sycl {

// Forward declaration
class device;

// The class device_selector is an abstract class which enables the SYCL runtime to choose the best device based
// on heuristics specified by the user, or by one of the built-in device selectors
struct device_selector {
	static std::unique_ptr<device_selector> default;
	virtual int operator()(device dev) = 0;
};

// TODO: Built-in device selectors
struct gpu_selector : device_selector {
	virtual int operator()(device dev) override;
};
struct cpu_selector : device_selector {
	virtual int operator()(device dev) override;
};
struct host_selector : device_selector {
	virtual int operator()(device dev) override;
};

} // namespace sycl
} // namespace cl
