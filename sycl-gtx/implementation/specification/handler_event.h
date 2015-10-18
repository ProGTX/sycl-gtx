#include "event.h"

namespace cl {
namespace sycl {

// TODO
class handler_event {
private:
	event kernel_;
	event complete_;
	event end_;

public:
	event get_kernel() const {
		return kernel_;
	}
	event get_complete() const {
		return complete_;
	}
	event get_end() const {
		return end_;
	}
};

} // namespace sycl
} // namespace cl
