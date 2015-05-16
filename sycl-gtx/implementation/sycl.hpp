#pragma once

#define CL_SYCL_LANGUAGE_VERSION 120

#include "specification\accessor\buffer.h"
#include "specification\accessor\local.h"
#include "specification\buffer.h"
#include "specification\command_group.h"
#include "specification\context.h"
#include "specification\device.h"
#include "specification\invoke.h"
#include "specification\kernel.h"
#include "specification\platform.h"
#include "specification\program.h"
#include "specification\queue.h"
#include "specification\ranges.h"
#include "specification\types.h"
#include "specification\workitem_functions.h"

#include "flow_control.h"

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
