#pragma once

#if _MSC_VER <= 1800
#define MSVC_LOW 1
#endif

// 3.2.0 VECTOR_CLASS and STRING_CLASS

#if __NO_STD_VECTOR
#define VECTOR_CLASS cl::sycl::vector
#elif __USE_DEV_VECTOR
#define VECTOR_CLASS std::string
#else
#include <vector>
#define VECTOR_CLASS std::vector
#endif

#if __NO_STD_STRING
#define STRING_CLASS cl::sycl::string
#elif __USE_DEV_STRING
#define STRING_CLASS std::string
#else
#include <string>
#define STRING_CLASS std::string
#endif
