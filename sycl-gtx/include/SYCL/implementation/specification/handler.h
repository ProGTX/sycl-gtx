#pragma once

// 3.5.3 SYCL functions for invoking kernels

#include "access.h"
#include "handler_event.h"
#include "program.h"
#include "ranges.h"
#include "../src_handlers/issue_command.h"
#include "../common.h"
#include "../function_traits.h"

namespace cl {
namespace sycl {

// Forward declarations
class kernel;
class queue;

class handler;
namespace detail {
static unique_ptr_class<handler> get_handler(queue* q);
}

// 3.5.3.4 Command group handler class
class handler {
private:
  friend class detail::command_group;
  friend unique_ptr_class<handler> detail::get_handler(queue* q);

  queue* q;
  handler_event events;

  // TODO: Implementation defined constructor
  handler(queue* q)
    : q(q) {}

  static context get_context(queue* q);

  template <class KernelType>
  shared_ptr_class<kernel> build(KernelType kernFunctor) {
    detail::command::group_::check_scope();
    program prog(get_context(q));
    prog.build(kernFunctor, "");

    // We know here the program only contains one kernel
    return prog.kernels.begin()->second;
  }

  using issue = detail::issue_command;

  template <class... Args>
  void issue_enqueue(
    shared_ptr_class<kernel> kern,
    void(*issue_enqueue_f)(shared_ptr_class<kernel>, event*, Args...),
    Args... params
  ) {
    issue::write_buffers_to_device(kern);
    issue_enqueue_f(kern, &events.kernel_, params...);
    issue::read_buffers_from_device(kern);
  }

public:
  // TODO
  template
    <typename DataType, int dimensions, access::mode mode, access::target target>
  void set_arg(int arg_index, accessor<DataType, dimensions, mode, target>& acc_obj);
  template <typename T>
  void set_arg(int arg_index, T scalar_value);


  // 3.5.3.1 Single Task invoke

  template <typename KernelName, class KernelType>
  void single_task(KernelType kernFunctor) {
    auto kern = build(kernFunctor);
    issue_enqueue(kern, &issue::enqueue_task);
  }


  // 3.5.3.2 Parallel For invoke

  template <typename KernelName, class KernelType, int dimensions>
  void parallel_for(range<dimensions> numWorkItems, KernelType kernFunctor) {
    parallel_for<KernelName>(numWorkItems, id<dimensions>(), kernFunctor);
  }

  // This type of kernel can be invoked with a function
  // accepting either an id or an item as parameter
  template <typename KernelName, class KernelType, int dimensions>
  void parallel_for(
    range<dimensions> numWorkItems,
    id<dimensions> workItemOffset,
    KernelType kernFunctor
  ) {
    auto kern = build(kernFunctor);
    issue_enqueue(kern, &issue::enqueue_range, numWorkItems, workItemOffset);
  }

  template <typename KernelName, class KernelType, int dimensions>
  void parallel_for(nd_range<dimensions> executionRange, KernelType kernFunctor) {
    auto kern = build(kernFunctor);
    issue_enqueue(kern, &issue::enqueue_nd_range, executionRange);
  }

  // TODO: Why is the offset needed? It's already contained in the nd_range
  template <typename KernelName, class KernelType, int dimensions>
  void parallel_for(
    nd_range<dimensions> numWorkItems,
    id<dimensions> workItemOffset,
    KernelType kernFunctor
  );


  // TODO: 3.5.3.3 Parallel For hierarchical invoke

  template <typename KernelName, class WorkgroupFunctionType, int dimensions>
  void parallel_for_work_group(
    range<dimensions> numWorkGroups, WorkgroupFunctionType kernFunctor);

  template <typename KernelName, class WorkgroupFunctionType, int dimensions>
  void parallel_for_work_group(
    range<dimensions> numWorkGroups,
    range<dimensions> workGroupSize,
    WorkgroupFunctionType kernFunctor
  );


  // Specializations for working with functors instead of lambdas

  template <class KernelType, class = decltype(KernelType::operator())>
  void single_task(KernelType kernFunctor) {
    single_task<KernelType, KernelType>(kernFunctor);
  }
  template
    <class KernelType, int dimensions, class = decltype(KernelType::operator())>
  void parallel_for(range<dimensions> numWorkItems, KernelType kernFunctor) {
    parallel_for<KernelType, KernelType, dimensions>(numWorkItems, kernFunctor);
  }
  template
    <class KernelType, int dimensions, class = decltype(KernelType::operator())>
  void parallel_for(
    range<dimensions> numWorkItems,
    id<dimensions> workItemOffset,
    KernelType kernFunctor
  ) {
    parallel_for<KernelType, KernelType, dimensions>(
      numWorkItems, workItemOffset, kernFunctor);
  }
  template <
    class KernelType,
    int dimensions,
    class = decltype(
      std::declval<KernelType>().operator()(std::declval<nd_item<1>>())
    )
  >
  void parallel_for(nd_range<dimensions> executionRange, KernelType kernFunctor) {
    parallel_for<KernelType, KernelType, dimensions>(executionRange, kernFunctor);
  }
  template
    <class KernelType, int dimensions, class = decltype(KernelType::operator())>
  void parallel_for(
    nd_range<dimensions> numWorkItems,
    id<dimensions> workItemOffset,
    KernelType kernFunctor
  ) {
    parallel_for<KernelType, KernelType, dimensions>(
      numWorkItems, workItemOffset, kernFunctor);
  }
  template <
    class WorkgroupFunctionType,
    int dimensions,
    class = decltype(WorkgroupFunctionType::operator())
  >
  void parallel_for_work_group(
    range<dimensions> numWorkGroups, WorkgroupFunctionType kernFunctor
  ) {
    parallel_for_work_group<WorkgroupFunctionType, WorkgroupFunctionType, dimensions>(
      numWorkGroups, kernFunctor);
  }
  template <
    class WorkgroupFunctionType,
    int dimensions,
    class = decltype(WorkgroupFunctionType::operator())
  >
  void parallel_for_work_group(
    range<dimensions> numWorkGroups,
    range<dimensions> workGroupSize,
    WorkgroupFunctionType kernFunctor
  ) {
    parallel_for_work_group<WorkgroupFunctionType, WorkgroupFunctionType, dimensions>(
      numWorkGroups, workGroupSize, kernFunctor);
  }


  // OpenCL interoperability invoke

  template <bool = true>
  void single_task(kernel syclKernel) {
    auto kern = shared_ptr_class<kernel>(new kernel(std::move(syclKernel)));
    issue_enqueue(kern, &issue::enqueue_task);
  }

  template <int dimensions>
  void parallel_for(range<dimensions> numWorkItems, kernel syclKernel) {
    auto kern = shared_ptr_class<kernel>(new kernel(std::move(syclKernel)));
    issue_enqueue(kern, &issue::enqueue_range, numWorkItems, id<dimensions>());
  }

  template <int dimensions>
  void parallel_for(nd_range<dimensions> ndRange, kernel syclKernel) {
    auto kern = shared_ptr_class<kernel>(new kernel(std::move(syclKernel)));
    issue_enqueue(kern, &issue::enqueue_nd_range, ndRange);
  }
};

namespace detail {
// Required for Clang
static unique_ptr_class<handler> get_handler(queue* q) {
  return unique_ptr_class<handler>(new handler(q));
}
}

} // namespace sycl
} // namespace cl
