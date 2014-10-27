#pragma once

// 3.3.4 Accessors

namespace cl {
namespace sycl {

namespace access {

// 3.3.4.1 Access modes
enum mode {
	read,				// read-only access
	write,				// write-only access, previous target object contents discarded
	atomic,				// atomic read and write access
	read_write,			// read/write access
	discard_read_write,	// read/write access, previous target object contents discarded
};

} // namespace access

} // namespace sycl
} // namespace cl
