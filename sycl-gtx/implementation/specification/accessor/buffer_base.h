#pragma once

// Core buffer accessor class

#include "../ranges.h"
#include "../../common.h"

namespace cl {
namespace sycl {
	
// Forward declaration
template <typename DataType, int dimensions>
struct buffer;

namespace detail {
	
template <typename DataType, int dimensions>
class accessor_buffer {
protected:
	buffer<DataType, dimensions>* buf;
	range<dimensions> offset;
	range<dimensions> rang;
public:
	accessor_buffer(
		buffer<DataType, dimensions>& bufferRef,
		range<dimensions> offset,
		range<dimensions> range
	) : buf(&bufferRef), offset(offset), rang(range) {
		DSELF() << "not implemented";
	}
protected:
	cl_mem get_buffer_object() const {
		return buf->device_data.get();
	}
	size_t access_buffer_range(int n) const {
		return buf->rang[n];
	}
	DataType* access_host_data() const {
		return buf->host_data.get();
	}
};

} // namespace detail
} // namespace sycl
} // namespace cl
