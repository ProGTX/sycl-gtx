#include "SYCL/detail/src_handlers/kernel_source.h"

#include "SYCL/access.h"
#include "SYCL/command_group.h"
#include "SYCL/error_handler.h"
#include "SYCL/kernel.h"
#include "SYCL/program.h"

using namespace cl::sycl;
using namespace detail::kernel_ns;

const string_class source::resource_name_root = "_sycl_buf";
SYCL_THREAD_LOCAL int source::num_resources = 0;
SYCL_THREAD_LOCAL source* source::scope = nullptr;

bool source::in_scope() {
  return scope != nullptr;
}

void source::enter(source& src) {
  scope = &src;
  num_resources = 0;
}

source source::exit(source& src) {
  scope = nullptr;
  return src;
}

/** Creates kernel source */
string_class source::get_code() const {
  // TODO(progtx): Caching?

  static const char newline = '\n';

  string_class final_code = string_class("__kernel void ") + kernel_name + "(" +
                            generate_accessor_list() + ") {" + newline;

  for (auto& line : lines) {
    final_code += line + newline;
  }

  final_code = final_code + "}" + newline;

  return final_code;
}

string_class source::get_kernel_name() const {
  return kernel_name;
}

string_class source::generate_accessor_list() const {
  string_class list;
  if (resources.empty()) {
    return list;
  }

  for (auto& acc : resources) {
    list += get_name(acc.second.acc.target) + " ";
    if (acc.second.acc.mode == access::mode::read) {
      list += "const ";
    }
    list += acc.second.type_name + " ";
    list += acc.second.resource_name + ", ";
  }

  // 2 to get rid of the last comma and space
  return list.substr(0, list.length() - 2);
}

string_class source::get_name(access::target target) {
  // TODO(progtx): All cases
  switch (target) {
    case access::target::global_buffer:
      return "__global";
    case access::target::constant_buffer:
      return "__constant";
    case access::target::local:
      return "__local";
    default:
      return "";
  }
}

void source::init_kernel(program& p, shared_ptr_class<kernel> kern) {
  ::cl_int error_code;
  cl_kernel k = clCreateKernel(p.get(), kernel_name.c_str(), &error_code);
  detail::error::report(error_code);
  kern->set(k);
  kern->kern.release_one();
}
