#pragma once

// 3.5.7 Event class

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
	// Constructs a copy sharing the same underlying event.
	// The underlying event is reference counted.
	event()
		: m_event(refc::allocate(clReleaseEvent)) {}

	cl_event get(cl_context context);
	vector_class<event> get_wait_list();
	void wait();
	static void wait(vector_class<event>& event_list);
	void wait_and_throw();
	static void wait_and_throw(const vector_class<event>& event_list);
};

} // namespace sycl
} // namespace cl
