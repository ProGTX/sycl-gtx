#pragma once

// TODO: 3.7.1.3 ID class

#include "range.h"
#include "../../common.h"
#include <initializer_list>

namespace cl {
namespace sycl {

template <int dimensions = 1>
class id : public range<dimensions> {
public:
	// TODO
	id(std::initializer_list<size_t> list)
		: range<dimensions>::range(vector_class<size_t>(dimensions, 0).data()) {}

	// TODO: This would be much easier if I could inherit constructors ...
	id(size_t sizeX)
		: range<dimensions>(sizeX) {}
};

} // namespace sycl
} // namespace cl
