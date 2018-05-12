#pragma once

// 3.5.6 Command group class

#include "SYCL/access.h"
#include "SYCL/buffer_base.h"
#include "SYCL/detail/common.h"
#include "SYCL/detail/debug.h"
#include "SYCL/ranges.h"
#include <set>

namespace cl {
namespace sycl {

// Forward declarations
class event;
class handler;
class kernel;
class queue;

namespace detail {

// Forward declarations
static inline unique_ptr_class<handler> get_handler(queue* q);
template <typename, int>
class buffer_detail;

namespace command {

// Forward declaration
class group_detail;

enum class type_t { unspecified, get_accessor, copy_data, kernel };

static debug& operator<<(debug& d, type_t t) {
  string_class str("command::type::");
  switch (t) {
    case type_t::get_accessor:
      str += "get_accessor";
      break;
    case type_t::copy_data:
      str += "copy_data";
      break;
    case type_t::kernel:
      str += "kernel";
      break;
    case type_t::unspecified:
    default:
      str += "unspecified";
      break;
  }
  d << str;
  return d;
}

struct buffer_copy {
  buffer_access buf;
  access::mode mode;
};

union metadata {
  std::nullptr_t empty;
  buffer_access buf_acc;
  buffer_copy buf_copy;

  metadata() : empty(nullptr) {}
  metadata(buffer_access buf_acc) : buf_acc(buf_acc) {}
  metadata(buffer_copy buf_copy) : buf_copy(buf_copy) {}
};

struct info {
  using command_f = function_class<void(queue*, const vector_class<cl_event>&)>;

  string_class name;  // Only for debugging
  command_f function;
  type_t type;
  metadata data;

  static void do_nothing(queue* q, const vector_class<cl_event>&) {}
};

}  // namespace command

/**
 * A command group in SYCL as it is defined in 2.3.1
 * includes a kernel to be enqueued along with all the commands
 * for queued data transfers that it needs in order for its execution to be
 * successful
 */
class command_group {
 private:
  friend class kernel;
  friend class command::group_detail;
  friend class ::cl::sycl::queue;
  using command_t = command::info;
  using command_f = command_t::command_f;

  vector_class<command_t> commands;
  std::set<buffer_base*> read_buffers;
  std::set<buffer_base*> write_buffers;
  queue* q;

  void enter();
  void exit();

 public:
  command_group(queue* q) : q(q) {}

  template <typename functorT>
  command_group(functorT lambda) : q(nullptr) {
    enter();
    auto cgh = get_handler(q);
    lambda(*cgh);
    exit();
  }

  /**
   * Constructs a command group with the queue the group will enqueue its
   * commands to and a lambda function or function object containing the
   * body of commands to enqueue.
   */
  template <typename functorT>
  command_group(queue& primaryQueue, functorT lambda) : q(&primaryQueue) {
    enter();
    auto cgh = get_handler(q);
    lambda(*cgh);
    exit();
  }

  // TODO(progtx):
  /**
   * Constructs a command group from a primary queue
   * to be used in order to enqueue its commands to
   * and a lambda function or function object
   * containing the body of commands to enqueue.
   * If the command group execution fails in the primary queue,
   * the SYCL runtime will try to re-schedule the whole command group
   * to the secondary queue.
   */
  template <typename functorT>
  command_group(queue& primaryQueue, queue& secondaryQueue, functorT lambda);

  void optimize();
  void flush(vector_class<cl_event> wait_events);
};

namespace command {

class group_detail {
 private:
  friend class ::cl::sycl::detail::command_group;

  // TODO(progtx): Need to deal better with threads
  SYCL_THREAD_LOCAL static command_group* last;

  template <class... Args>
  using fn = void (*)(queue*, const vector_class<cl_event>&, Args...);

  template <class... Args>
  using kern_fn = fn<shared_ptr_class<kernel>, event*, Args...>;

  template <type_t type = type_t::unspecified, class F, class... Args>
  static void add_command(F function, string_class name, Args... params) {
    last->commands.push_back({name,
                              std::bind(function, std::placeholders::_1,
                                        std::placeholders::_2, params...),
                              type});
  }

 public:
  static void add_kernel_enqueue_task(kern_fn<> function, string_class name,
                                      shared_ptr_class<kernel> kern,
                                      event* evnt) {
    add_command(function, name, kern, evnt);
  }

  template <int dimensions>
  static void add_kernel_enqueue_range(
      kern_fn<range<dimensions>, id<dimensions>> function, string_class name,
      shared_ptr_class<kernel> kern, event* evnt,
      range<dimensions> num_work_items, id<dimensions> offset) {
    add_command(function, name, kern, evnt, num_work_items, offset);
  }

  template <int dimensions>
  static void add_kernel_enqueue_nd_range(
      kern_fn<nd_range<dimensions>> function, string_class name,
      shared_ptr_class<kernel> kern, event* evnt,
      nd_range<dimensions> execution_range) {
    add_command(function, name, kern, evnt, execution_range);
  }

  template <typename DataType, int dimensions>
  static void add_buffer_init(fn<buffer_detail<DataType, dimensions>*> function,
                              string_class name,
                              buffer_detail<DataType, dimensions>* buff) {
    add_command(function, name, buff);
  }

  static void add_buffer_access(buffer_access buf_acc, string_class name);

  static void add_buffer_copy(
      buffer_access buf_acc, access::mode copy_mode,
      fn<buffer_base*, buffer_base::clEnqueueBuffer_f> function,
      string_class name, buffer_base* buffer,
      buffer_base::clEnqueueBuffer_f enqueue_function);

  static bool in_scope();
  static void check_scope();

  using command_f = info::command_f;
};

}  // namespace command

}  // namespace detail

}  // namespace sycl
}  // namespace cl
