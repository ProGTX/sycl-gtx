#include "command_group.h"

using namespace cl::sycl;

void command_group::enter() {
	detail::cmd_group::last = this;
}
void command_group::exit() {
	detail::cmd_group::last = nullptr;
}


using namespace detail;

command_group* cmd_group::last = nullptr;

void cmd_group::flush(queue* q) {
	for(auto&& command : last->commands) {
		command(q);
	}
	last->commands.clear();
}

bool cmd_group::in_scope() {
	return last != nullptr;
}
