# sycl-gtx

Implementation of the SYCL specification.
Learn more about SYCL [from the Khronos website](https://www.khronos.org/sycl).

This implementation is built on top of OpenCL 1.2
and as such it can work with any device that supports OpenCL 1.2 or higher
\- it relies on the OpenCL driver to compile the kernels.

The project started as a Masters project
([link to the thesis](http://eprints.fri.uni-lj.si/3292/1/%C5%BDu%C5%BEek.pdf),
the front page is in Slovenian but the majority is in English)
and was presented as a paper at the
[1st SYCL Programming Workshop](http://ppopp16.sigplan.org/event/sycl-2016-papers-an-overview-of-sycl-gtx)
at PPoPP 2016.

## Installation

### Requirements

* CMake 3.2
* Compiler with full C++11 support
  * Known to successfully build on Ubuntu 16.04
    using GCC 4.8 and 5.4 or Clang versions 3.5, 3.6, and 3.8
  * Visual Studio 2013 has been used for development,
    but is not supported anymore
  * Visual Studio 2017, although 2015 should still work
* OpenCL

### Steps

```
git clone https://github.com/ProGTX/sycl-gtx.git
cd sycl-gtx
mkdir build
cd build
cmake ..
```

Ideally, that's all that's needed to generate a
Visual Studio solution or a makefile.

### OpenCL setup

## Kernel compilation

A very important part of sycl-gtx is the way it compiles kernels.
A kernel executes on an OpenCL device,
but since sycl-gtx can be used with any host compiler
and the kernel is written in normal C++11 code,
the compiler has no idea it's compiling code for an OpenCL device.
sycl-gtx tries to work around this in a few ways:
1. Non-standard macros used for control flow,
   e.g. `SYCL_IF` and `SYCL_END`
1. Using the non-standard `data_ref` class to store variable information
   \- this is done as an embedded DSL using overloaded operators.
1. Some extra types to aid the embedded DSL,
   e.g. `cl::sycl::int1` should be used instead of `int`.
1. Hacks for copying data to the device.

In many simple cases, the embedded DSL will take care of everything
and the code would have no trouble compiling on another SYCL implementation.
In some cases, some extra types and macros need to be used in the kernel
and this project provides the file `CL/sycl_gtx_compatibility.hpp`
that enables this specially decorated code
to be compatible with other implementations.

Unfortunately, moving custom data types complicates things further.
SYCL allows custom classes to be used in the kernel,
but there is no simple solution to do that in sycl-gtx,
so some code refactoring may be required.
This repository provides the smallpt project,
where the [smallpt ray tracer](http://www.kevinbeason.com/smallpt/)
was ported to sycl-gtx
and it illustrates the changes required to make it work.

The embedded DSL collects information on the types and values
and creates string representations of them,
partially at compile time and partially at runtime.
The kernel is then transformed into a string at runtime
and passed to `clCreateProgramFromSource`.

## Current Status

At the moment, the implementation is far from complete,
though it does cover quite a lot of the SYCL 1.2 specification
and it's able to compile many simple programs.

The implementation is being improved, albeit slowly
\- a single developer can do only so much.
Reporting issues is very welcome as it helps find areas that need work most.

SYCL provides a host device,
but implementing that in sycl-gtx would require a lot of work,
so it's not planned anytime soon.
A good start would be to carry values inside the `data_ref` class
and to make them type safe.

## The SYCL ecosystem

There are other known implementations of SYCL,
like [ComputeCpp](https://www.codeplay.com/products/computesuite/computecpp),
[triSYCL](https://github.com/keryell/triSYCL),
[hipSYCL](https://github.com/illuhad/hipSYCL),
and [Intel LLVM SYCL](https://github.com/intel/llvm/tree/sycl).

In addition to implementations,
there is an effort to build the SYCL ecosystem
by creating tools and libraries using SYCL
so that developers can then use tools they already know,
just with SYCL running underneath.
sycl-gtx will try to cover at least some of those efforts.

A good ecosystem starting point is [sycl.tech](https://sycl.tech/),
along with [SyclParallelSTL](https://github.com/KhronosGroup/SyclParallelSTL)
(an implementation of the C++17 Parallel STL in SYCL),
[ComputeCpp SDK](https://github.com/codeplaysoftware/computecpp-sdk)
(some sample code written in SYCL),
[VisionCpp](https://github.com/codeplaysoftware/visioncpp)
(a library for computer vision and image processing).

## License

The project is licensed under MIT
\- see the LICENSE file which applies to all code files.
