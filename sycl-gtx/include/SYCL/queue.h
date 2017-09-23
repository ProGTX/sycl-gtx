#pragma once

// 3.3.5 Queue class

#include "SYCL/command_group.h"
#include "SYCL/context.h"
#include "SYCL/detail/common.h"
#include "SYCL/detail/debug.h"
#include "SYCL/detail/synchronizer.h"
#include "SYCL/device.h"
#include "SYCL/error_handler.h"
#include "SYCL/handler_event.h"
#include "SYCL/info.h"
#include "SYCL/param_traits.h"
#include "SYCL/refc.h"

namespace cl {
namespace sycl {

/** Encapsulation of an OpenCL cl_command_queue */
class queue {
 private:
  friend class detail::synchronizer;

  using buffer_set = std::set<detail::buffer_base*>;

  context ctx;
  device dev;
  detail::refc<cl_command_queue, clRetainCommandQueue, clReleaseCommandQueue>
      command_q;
  exception_list ex_list;
  detail::command_group command_group;
  buffer_set buffers_in_use;
  bool is_flushed = true;
  vector_class<queue> subqueues;

  void display_device_info() const;
  cl_command_queue create_queue(bool display_info = true,
                                bool register_with_synchronizer = true,
                                info::queue_profiling enable_profiling = false);

 public:
  /**
   * Creates a queue for a device it chooses
   * according to the heuristics of the default selector.
   * The OpenCL context object is created implicitly.
   */
  explicit queue(
      const async_handler& asyncHandler = detail::default_async_handler);

  queue(const device_selector& deviceSelector,
        const async_handler& asyncHandler = detail::default_async_handler);

  queue(const context& syclContext, const device_selector& deviceSelector,
        const async_handler& asyncHandler = detail::default_async_handler);

  queue(const context& syclContext, const device& syclDevice,
        const async_handler& asyncHandler = detail::default_async_handler);

  /** Chooses a device based on the provided device selector in the given
   * context.
   */
  queue(const context& syclContext, const device& syclDevice,
        info::queue_profiling profilingFlag,
        const async_handler& asyncHandler = detail::default_async_handler);

  /** Creates a queue for the provided device. */
  queue(const device& syclDevice,
        const async_handler& asyncHandler = detail::default_async_handler);

  /**
   * Creates a SYCL queue from an OpenCL queue.
   * At construction it does a retain on the queue memory object.
   */
  queue(cl_command_queue clQueue,
        const async_handler& asyncHandler = detail::default_async_handler);

 private:
  /** Create sub-queue, which executes the command group immediately */
  template <typename T>
  queue(queue* master, T cgf)
      : ctx(master->ctx),
        dev(master->dev),
        command_q(create_queue(false, false)),
        command_group(*this, cgf),
        is_flushed(false) {}

 public:
  ~queue();

  // Copy semantics
  queue(const queue&) = default;
  queue& operator=(const queue&) = default;

  /**
   * Queue requires custom move semantics
   * because the parent pointer is carrier in subqueues
   */
  queue(queue&& move) noexcept
      : SYCL_MOVE_INIT(ctx),
        SYCL_MOVE_INIT(dev),
        SYCL_MOVE_INIT(command_q),
        SYCL_MOVE_INIT(ex_list),
        SYCL_MOVE_INIT(command_group),
        SYCL_MOVE_INIT(buffers_in_use),
        SYCL_MOVE_INIT(is_flushed),
        SYCL_MOVE_INIT(subqueues) {
    move.command_q = nullptr;
    command_group.q = this;
  }
  queue& operator=(queue&& move) noexcept {
    std::swap(*this, move);
    return *this;
  }
  friend void swap(queue& first, queue& second) {
    using std::swap;
    SYCL_SWAP(ctx);
    SYCL_SWAP(dev);
    SYCL_SWAP(command_q);
    SYCL_SWAP(ex_list);
    SYCL_SWAP(command_group);
    SYCL_SWAP(buffers_in_use);
    SYCL_SWAP(is_flushed);
    SYCL_SWAP(subqueues);
  }

  bool is_host();

  // TODO(progtx): Returns the underlying OpenCL command queue after doing a
  // retain.
  /** Afterwards it needs to be manually released. */
  cl_command_queue get();

  /** Returns the SYCL context the queue is using. */
  context get_context() const;

  /** Returns the SYCL device the queue is associated with. */
  device get_device() const;

  template <info::queue param>
  typename param_traits<info::queue, param>::type get_info() const {
    return detail::non_vector_traits<info::queue, param, 1>().get(
        command_q.get());
  }

  /**
   * Checks to see if any asynchronous errors have been produced by the queue
   * and if so reports them by passing them to the async_handler
   * provided on construction.
   * If no async_handler was provided then asynchronous exceptions will be lost.
   */
  void throw_asynchronous();

  /**
   * Performs a blocking wait for the completion all enqueued tasks in the
   * queue.
   * Synchronous errors will be reported via an exception.
   */
  void wait();

  /** Performs a blocking wait for the completion of all enqueued tasks in the
   * queue.
   */
  void wait_and_throw();

  // TODO(progtx):
  template <typename T>
  handler_event submit(T cgf) {
    subqueues.push_back({this, cgf});
    return subqueues.back().process(buffers_in_use);
  }

  // TODO(progtx):
  template <typename T>
  handler_event submit(T cgf, queue& secondaryQueue);

 private:
  void flush();
  void finish();
  void wait_subqueues(bool and_throw);
  handler_event process(buffer_set& buffers_in_use_master);
  static vector_class<cl_event> get_wait_events(const buffer_set& dependencies,
                                                buffer_set& buffers_in_use);
};

}  // namespace sycl
}  // namespace cl
