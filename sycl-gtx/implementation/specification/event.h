#pragma once

// 3.2.7 Event class

#include "error_handler.h"
#include "refc.h"
#include "../common.h"
#include "../debug.h"

namespace cl {
namespace sycl {

class event {
private:
	refc::ptr<cl_event> m_event;

public:
	event(cl_event from_cl_event = nullptr)
		: m_event(refc::allocate<>(from_cl_event, clReleaseEvent))
	{
		if(from_cl_event != nullptr) {
			auto error_code = clRetainEvent(from_cl_event);
			detail::error::handler().report(error_code);
		}
	}

	cl_event get(cl_context context);
	static void wait(vector_class<event> event_list);
	vector_class<event> get_wait_list();
};

} // namespace sycl
} // namespace cl
