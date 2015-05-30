#include "event.h"

using namespace cl::sycl;

event::event(cl_event clEvent)
	: evnt(clEvent) {
	auto error_code = clRetainEvent(clEvent);
	detail::error::report(error_code);
}

cl_event event::get() {
	return evnt.get();
}

vector_class<event> event::get_wait_list() {
	// TODO
	return {};
}

void event::wait() {
	auto ev = evnt.get();
	auto error_code = clWaitForEvents(1, &ev);
	detail::error::report(error_code);
}

void event::wait(const vector_class<event>& event_list) {
	vector_class<cl_event> events;
	auto size = event_list.size();
	events.reserve(size);
	for(auto& e : event_list) {
		events.emplace_back(e.evnt.get());
	}

	auto error_code = clWaitForEvents(size, events.data());
	detail::error::report(error_code);
}

void event::wait_and_throw() {
	// TODO
}
void event::wait_and_throw(const vector_class<event>& event_list) {
	// TODO
}
