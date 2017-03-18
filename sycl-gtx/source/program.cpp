#include "SYCL/program.h"

#include "SYCL/detail/debug.h"
#include "SYCL/kernel.h"
#include "SYCL/queue.h"

using namespace cl::sycl;

program::program(cl_program clProgram, const context& context,
                 vector_class<device> deviceList)
    : prog(clProgram), ctx(context), devices(deviceList) {}

program::program(const context& context)
    : program(context, context.get_devices()) {}

program::program(const context& context, vector_class<device> deviceList)
    : program(nullptr, context, deviceList) {}

program::program(const context& context, cl_program clProgram)
    : program(clProgram, context, context.get_devices()) {}

void program::compile(string_class compile_options, ::size_t kernel_name_id,
                      shared_ptr_class<kernel> kern) {
  kernels.emplace(kernel_name_id, kern);
  auto& src = kern->src;
  auto code = src.get_code();

  debug() << "Compiled kernel:";
  debug() << code;

  const char* code_p = code.c_str();
  ::size_t length = code.size();
  ::cl_int error_code;

  auto p =
      clCreateProgramWithSource(ctx.get(), 1, &code_p, &length, &error_code);
  detail::error::report(error_code);
  kern->set(ctx, p);
  kern->prog->prog.release_one();

  auto device_pointers = detail::get_cl_array(devices);

  error_code = clCompileProgram(kern->prog.get()->get(),
                                static_cast<::cl_uint>(devices.size()),
                                device_pointers.data(), compile_options.c_str(),
                                0, nullptr, nullptr, nullptr, nullptr);

  try {
    detail::error::report(error_code);
  } catch (::cl::sycl::exception& e) {
    debug() << "Error while compiling kernel" << kern->src.get_kernel_name()
            << "->";
    for (auto& d : devices) {
      report_compile_error(kern, d);
    }
    throw e;
  }
}

void program::report_compile_error(shared_ptr_class<kernel> kern,
                                   device& dev) const {
  // http://stackoverflow.com/a/9467325/793006

  // Determine the size of the log
  ::size_t log_size;
  clGetProgramBuildInfo(kern->prog.get()->get(), dev.get(),
                        CL_PROGRAM_BUILD_LOG, 0, nullptr, &log_size);

  // Allocate memory for the log
  auto log = new char[log_size];

  // Get the log
  clGetProgramBuildInfo(kern->prog.get()->get(), dev.get(),
                        CL_PROGRAM_BUILD_LOG, log_size, log, nullptr);

  debug() << "\tWhile compiling for device"
          << dev.get_info<info::device::name>() << "->\n"
          << log;

  delete[] log;
}

void program::init_kernels() {
  for (auto& kern : kernels) {
    // The extra kernel parameter is required because of complex dependencies
    kern.second->src.init_kernel(*this, kern.second);
  }
}

vector_class<cl_program> program::get_program_pointers() const {
  vector_class<cl_program> program_pointers;
  program_pointers.reserve(kernels.size());

  for (auto& kern : kernels) {
    program_pointers.push_back(kern.second->prog.get()->get());
  }

  return program_pointers;
}

void program::link(string_class linking_options) {
  if (linked) {
    // TODO(progtx): Error?
    return;
  }

  auto device_pointers = detail::get_cl_array(devices);
  auto program_pointers = get_program_pointers();
  ::cl_int error_code;

  prog =
      clLinkProgram(ctx.get(), static_cast<::cl_uint>(device_pointers.size()),
                    device_pointers.data(), linking_options.c_str(),
                    static_cast<::cl_uint>(program_pointers.size()),
                    program_pointers.data(), nullptr, nullptr, &error_code);
  detail::error::report(error_code);

  // Can only initialize after program successfully built
  init_kernels();

  linked = true;
}
