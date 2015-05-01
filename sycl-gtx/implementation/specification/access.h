#pragma once

#include "../debug.h"

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

static debug& operator<<(debug& d, mode m) {
	d << "mode::";
	switch(m) {
		case read:
			d << "read";
			break;
		case write:
			d << "write";
			break;
		case atomic:
			d << "atomic";
			break;
		case read_write:
			d << "read_write";
			break;
		case discard_read_write:
			d << "discard_read_write";
			break;
	}
	return d;
}

static debug& operator<<(debug& d, target t) {
	d << "target::";
	switch(t) {
		case global_buffer:
			d << "global_buffer";
			break;
		case constant_buffer:
			d << "constant_buffer";
			break;
		case local:
			d << "local";
			break;
		case image:
			d << "image";
			break;
		case host_buffer:
			d << "host_buffer";
			break;
		case host_image:
			d << "host_image";
			break;
		case image_array:
			d << "image_array";
			break;
		case cl_buffer:
			d << "cl_buffer";
			break;
		case cl_image:
			d << "cl_image";
			break;
	}
	return d;
}

} // namespace access
} // namespace sycl
} // namespace cl
