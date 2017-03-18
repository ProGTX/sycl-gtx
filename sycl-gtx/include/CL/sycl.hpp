#pragma once

#define SYCL_GTX
#define CL_SYCL_LANGUAGE_VERSION 120

#include "SYCL/accessors/buffer.h"
#include "SYCL/accessors/local.h"
#include "SYCL/buffer.h"
#include "SYCL/command_group.h"
#include "SYCL/context.h"
#include "SYCL/device.h"
#include "SYCL/functions/common.h"
#include "SYCL/handler.h"
#include "SYCL/info.h"
#include "SYCL/kernel.h"
#include "SYCL/platform.h"
#include "SYCL/program.h"
#include "SYCL/queue.h"
#include "SYCL/ranges.h"
#include "SYCL/vectors/swizzled_vec.h"
#include "SYCL/vectors/vec.h"
#include "SYCL/workitem_functions.h"

#include "SYCL/detail/flow_control.h"

#if MSVC_2013_OR_LOWER
#undef MSVC_2013_OR_LOWER
#endif

#undef SYCL_ACCESSOR_CLASS
#undef SYCL_ADD_ACCESSOR
#undef SYCL_DEVICE_REF_SUBSCRIPT_OP
#undef SYCL_DEVICE_REF_SUBSCRIPT_OPERATORS
#undef SYCL_MOVE_INIT
#undef SYCL_THREAD_LOCAL
#undef SYCL_SWAP
