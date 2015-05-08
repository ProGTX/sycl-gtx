#include "command_group.h"
#include <unordered_map>

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

	std::unordered_map<command_t*, bool> keep(commands.size());

	for(auto& command : commands) {
		keep[&command] = true;
	}

	decltype(commands) new_commands;
	new_commands.reserve(commands.size());

	for(auto& command : commands) {
		if(keep[&command]) {
			new_commands.push_back(std::move(command));
		}
	}

	commands = std::move(new_commands);
}

// Executes all commands in queue and removes them
void command_group::flush() {
	DSELF();

	optimize();

	using detail::command::type_t;

	for(auto& command : commands) {
		if(command.type == type_t::get_accessor) {
			auto& acc = command.data.buf_acc;
			auto d = debug();
			d << command.type << acc.buffer << acc.mode << acc.target;
		}
		else if(command.type == type_t::copy_data) {
			auto& copy = command.data.buf_copy;
			auto d = debug();
			d << command.type << copy.buf.buffer << copy.buf.mode << copy.buf.target << copy.mode;
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
