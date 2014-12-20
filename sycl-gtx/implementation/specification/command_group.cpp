#include "command_group.h"

using namespace cl::sycl;

void command_group::enter() {
	detail::cmd_group::last = this;
}
void command_group::exit() {
	// TODO: Move flush to end of caller queue
	detail::cmd_group::flush();
	detail::cmd_group::last = nullptr;
}


using namespace detail;

command_group* cmd_group::last = nullptr;

void cmd_group::flush() {
	DSELF();

	for(auto&& command : last->commands) {
		command(last->q);
	}
	last->commands.clear();
}

bool cmd_group::in_scope() {
	return last != nullptr;
}
