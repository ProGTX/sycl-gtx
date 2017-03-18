#include "../common.h"

// 2.5.6 Anatomy of a SYCL application
// (slightly revised from revision 2014-03-09 of the specification)

// Originally test2

int main() {
  using namespace cl::sycl;

  static const int expected = 1024;

  int result;  // this is where we will write our result

  {  // by sticking all the SYCL work in a {} block, we ensure
    // all SYCL tasks must complete before exiting the block

    // create a queue to work on
    queue myQueue;

    // wrap our result variable in a buffer
    buffer<int> resultBuf(&result, 1);

    // create some commands for our queue
    myQueue.submit([&](handler& cgh) {
      // request access to our buffer
      auto writeResult = resultBuf.get_access<access::mode::write>(cgh);

      // enqueue a single, simple task
      cgh.single_task<class simple_test>([=]() { writeResult[0] = expected; });
    });  // end of our commands for this queue

  }  // end scope, so we wait for the queue to complete

  debug() << "Expected result" << expected;
  debug() << "Actual result" << result;

  return static_cast<int>(result != expected);
}
