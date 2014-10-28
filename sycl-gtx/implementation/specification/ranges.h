#pragma once

// 3.4.1 Ranges and identifiers


namespace cl {
namespace sycl {

template <int dimensions>
class range {};

template <int dimensions>
class nd_range {};

template <int dimensions = 1>
class id {
public:
	id(range<dimensions> global_size, range<dimensions> local_size) {}
	
	// TODO: Not in specification
	id(int size) {}

	int get(int dimension) {}
};


} // namespace sycl
} // namespace cl
