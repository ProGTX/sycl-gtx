#pragma once

#define SYCL_GTX
#define CL_SYCL_LANGUAGE_VERSION 120

#include "accessor/buffer.h"
#include "accessor/local.h"
#include "buffer.h"
#include "command_group.h"
#include "context.h"
#include "device.h"
#include "functions/common.h"
#include "handler.h"
#include "info.h"
#include "kernel.h"
#include "platform.h"
#include "program.h"
#include "queue.h"
#include "ranges.h"
#include "vectors/swizzled_vec.h"
#include "vectors/vec.h"
#include "workitem_functions.h"

#include "detail/flow_control.h"

#if MSVC_LOW
#undef MSVC_LOW
#endif

#undef SYCL_ACCESSOR_CLASS
#undef SYCL_ADD_ACCESSOR
#undef SYCL_DEVICE_REF_SUBSCRIPT_OP
#undef SYCL_DEVICE_REF_SUBSCRIPT_OPERATORS
#undef SYCL_MOVE_INIT
#undef SYCL_THREAD_LOCAL
#undef SYCL_SWAP
