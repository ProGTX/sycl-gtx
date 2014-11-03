#include "device_selector.h"
#include "device.h"

using namespace cl::sycl;

std::unique_ptr<device_selector> device_selector::default = std::unique_ptr<device_selector>(new host_selector());

int host_selector::operator()(device dev) {
	DSELF() << "not implemented";
	return 0;
}
