#include "command_group.h"

using namespace cl::sycl;

void command_group::enter() {
	detail::command::group_::last = this;
}
void command_group::exit() {
	detail::command::group_::last = nullptr;
}

void command_group::flush() {
	DSELF();

	for(auto& command : commands) {
		debug() << "command:" << command.name;
		command.function(q);
	}
	commands.clear();
}

command_group::~command_group() {
	// TODO: Move flush to end of caller command queue
	flush();
}


using namespace detail;

command_group* command::group_::last = nullptr;

bool command::group_::in_scope() {
	return last != nullptr;
}

void command::group_::check_scope(error::handler& handler) {
	if(!in_scope()) {
		handler.report(error::code::NOT_IN_COMMAND_GROUP_SCOPE);
	}
}
