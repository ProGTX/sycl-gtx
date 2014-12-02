#include "command_group.h"

using namespace cl::sycl;

command_group* detail::command_group_::last = nullptr;
