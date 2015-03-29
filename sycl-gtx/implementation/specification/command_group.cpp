#include "command_group.h"

using namespace cl::sycl;

void command_group::enter() {
	detail::cmd_group::last = this;
}
void command_group::exit() {
	detail::cmd_group::last = nullptr;
}

void command_group::flush() {
	DSELF();

	for(auto& command : commands) {
		debug() << "command:" << command.first;
		command.second(q);
	}
	commands.clear();
}

command_group::~command_group() {
	// TODO: Move flush to end of caller command queue
	flush();
}


using namespace detail;

command_group* cmd_group::last = nullptr;

bool cmd_group::in_scope() {
	return last != nullptr;
}

void cmd_group::check_scope(error::handler& handler) {
	if(!in_scope()) {
		handler.report(error::code::NOT_IN_COMMAND_GROUP_SCOPE);
	}
}
