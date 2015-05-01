#include "command_group.h"

using namespace cl::sycl;

void command_group::enter() {
	detail::command::group_::last = this;
}
void command_group::exit() {
	detail::command::group_::last = nullptr;
}

// TODO: Reschedules commands to achieve better performance
void command_group::optimize() {
	DSELF();
}

// Executes all commands in queue and removes them
void command_group::flush() {
	DSELF();

	optimize();

	using detail::command::type_t;

	for(auto& command : commands) {
		if(command.type == type_t::copy_data || command.type == type_t::get_accessor) {
			auto& acc = command.data.buf_acc;
			auto d = debug();
			d << command.type << acc.buffer << acc.mode << acc.target;
		}
		else {
			debug() << "command:" << command.name;
		}
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
