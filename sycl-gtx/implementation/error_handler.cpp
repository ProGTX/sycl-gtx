#include "error_handler.h"

using namespace cl::sycl;

helper::err_handler::type helper::err_handler::default_handler = helper::err_handler::type();

helper::err_handler::err_handler(type& handler)
	: handler(handler) {}
helper::err_handler::err_handler()
	: err_handler(default_handler) {}

#if MSVC_LOW
helper::err_handler::err_handler(err_handler&& move)
	: handler(move.handler) {}
helper::err_handler& helper::err_handler::operator=(err_handler&& move) {
	std::swap(handler, move.handler);
	return *this;
}
#endif

void helper::err_handler::handle(cl_int error_code, type& handler) {
#ifdef __CL_ENABLE_EXCEPTIONS
	DSELF() << "not implemented";
#else
	handler = error_code;
#endif
}

void helper::err_handler::handle(cl_int error_code) {
	handle(error_code, handler);
}
