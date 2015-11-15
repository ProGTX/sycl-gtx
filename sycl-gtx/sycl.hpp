#pragma once

#define SYCL_GTX
#define CL_SYCL_LANGUAGE_VERSION 120

#include "implementation/specification/accessor/buffer.h"
#include "implementation/specification/accessor/local.h"
#include "implementation/specification/buffer.h"
#include "implementation/specification/command_group.h"
#include "implementation/specification/context.h"
#include "implementation/specification/device.h"
#include "implementation/specification/handler.h"
#include "implementation/specification/info.h"
#include "implementation/specification/kernel.h"
#include "implementation/specification/platform.h"
#include "implementation/specification/program.h"
#include "implementation/specification/queue.h"
#include "implementation/specification/ranges.h"
#include "implementation/specification/types.h"
#include "implementation/specification/workitem_functions.h"

#include "implementation/flow_control.h"

#if MSVC_LOW
#undef MSVC_LOW
#undef SYCL_SWAP
#undef SYCL_MOVE_INIT
#endif

#undef SYCL_ACCESSOR_CLASS
#undef SYCL_ADD_ACCESSOR
#undef SYCL_DEVICE_REF_SUBSCRIPT_OP
#undef SYCL_DEVICE_REF_SUBSCRIPT_OPERATORS
#undef SYCL_THREAD_LOCAL
