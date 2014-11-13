#pragma once

// 3.4.1 Ranges and identifiers

#include "../debug.h"

namespace cl {
namespace sycl {

template <int dimensions = 1>
class range {
public:
	range() {
		DSELF() << "not implemented";
	}
};

template <>
class range<1> {
public:
	range() {
		DSELF() << "not implemented";
	}
	range(int) {
		DSELF() << "not implemented";
	}
};

template <int dimensions>
class nd_range {
public:
	nd_range() {
		DSELF() << "not implemented";
	}
};

template <int dimensions = 1>
class id {
public:
	id(range<dimensions> global_size, range<dimensions> local_size) {
		DSELF() << "not implemented";
	}

	int get(int dimension) {
		DSELF() << "not implemented";
	}
};


} // namespace sycl
} // namespace cl
