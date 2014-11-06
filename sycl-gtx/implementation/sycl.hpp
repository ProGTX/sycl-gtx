#pragma once

// TODO
#define CL_SYCL_LANGUAGE_VERSION

#include "specification\accessor.h"
#include "specification\buffer.h"
#include "specification\command_group.h"
#include "specification\context.h"
#include "specification\device.h"
#include "specification\kernel.h"
#include "specification\platform.h"
#include "specification\program.h"
#include "specification\queue.h"
#include "specification\ranges.h"

// Can collide with cl.hpp
#undef VECTOR_CLASS
#undef STRING_CLASS

#if MSVC_LOW
#undef SYCL_SWAP
#undef SYCL_MOVE_INIT
#endif
