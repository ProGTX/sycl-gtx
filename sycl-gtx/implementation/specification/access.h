#pragma once

namespace cl {
namespace sycl {
namespace access {

// 3.6.4.1 Access modes
enum mode {
	read,				// read-only access
	write,				// write-only access, previous target object contents discarded
	atomic,				// atomic read and write access
	read_write,			// read/write access
	discard_read_write,	// read/write access, previous target object contents discarded
};

// 3.6.4.2 Access targets
enum target {
	global_buffer = 0,	// access buffer via __global memory
	constant_buffer,	// access buffer via __constant memory
	local,				// access work-group - local memory
	image,				// access an image
	host_buffer,		// access buffer immediately on the host
	host_image,			// access image immediately on the host
	image_array,		// access an array of images on device
	cl_buffer,			// access an OpenCL cl_mem buffer on device
	cl_image,			// access an OpenCL cl_mem image on device
};

} // namespace access
} // namespace sycl
} // namespace cl
