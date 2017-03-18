#pragma once

#ifdef _MSC_VER
#if _MSC_VER <= 1910
#define MSVC_2017_OR_LOWER 1
#if _MSC_VER <= 1900
#define MSVC_2015_OR_LOWER 1
#if _MSC_VER <= 1800
#define MSVC_2013_OR_LOWER 1
#endif  // 1800
#endif  // 1900
#endif  // 1910
#endif  // _MSC_VER
