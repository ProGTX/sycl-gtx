
#ifndef SYCL_GTX

namespace cl {
namespace sycl {

#define SYCL_TYPE_ONE(type) using type##1 = type;

SYCL_TYPE_ONE(bool)

#define SYCL_TYPE_VEC(type)     \
  SYCL_TYPE_ONE(type)           \
  using type##2 = cl_##type##2; \
  using type##3 = cl_##type##3; \
  using type##4 = cl_##type##4; \
  using type##8 = cl_##type##8; \
  using type##16 = cl_##type##16;

SYCL_TYPE_ONE(float)
SYCL_TYPE_ONE(double)

#define SYCL_UTYPE_ONE(type) \
  SYCL_TYPE_ONE(type);       \
  using u##type##1 = unsigned type;

#define SYCL_UTYPE_VEC(type)    \
  SYCL_UTYPE_ONE(type)          \
  using type##2 = cl_##type##2; \
  using type##3 = cl_##type##3; \
  using type##4 = cl_##type##4; \
  using type##8 = cl_##type##8; \
  using type##16 = cl_##type##16;

SYCL_UTYPE_ONE(int)
SYCL_UTYPE_ONE(char)
SYCL_UTYPE_ONE(short)
SYCL_UTYPE_ONE(long)

#undef SYCL_TYPE_ONE
#undef SYCL_TYPE_VEC
#undef SYCL_UTYPE_ONE
#undef SYCL_UTYPE_VEC

}  // namespace sycl
}  // namespace cl

#define SYCL_BEGIN {
#define SYCL_END }

#define SYCL_IF(condition) \
  if (condition)           \
  SYCL_BEGIN

#define SYCL_ELSE SYCL_END else SYCL_BEGIN
#define SYCL_ELSE_IF(condition) SYCL_END else if (condition) SYCL_BEGIN

#define SYCL_WHILE(condition) \
  while (condition)           \
  SYCL_BEGIN
#define SYCL_FOR(init, condition, increment) \
  for (init; condition; increment)           \
  SYCL_BEGIN

#define SYCL_BREAK break;
#define SYCL_CONTINUE continue;
#define SYCL_RETURN return;

#endif  // SYCL_GTX
