
#ifndef SYCL_GTX

namespace cl {
namespace sycl {

#define SYCL_TYPE(type)	\
using type##1 = type;

SYCL_TYPE(bool)
SYCL_TYPE(float)
SYCL_TYPE(double)

#undef SYCL_TYPE


#define SYCL_TYPE(type)	\
using type##1 = type;	\
using u##type##1 = unsigned type;

SYCL_TYPE(int)
SYCL_TYPE(char)
SYCL_TYPE(short)
SYCL_TYPE(long)

#undef SYCL_TYPE

} // namespace sycl
} // namespace cl


#define SYCL_IF(condition) if(condition)

#define SYCL_BEGIN {

#define SYCL_END }

#define SYCL_BLOCK(code)	\
{							\
code						\
}

#define SYCL_THEN(code) \
SYCL_BLOCK(code)

#define SYCL_ELSE else

#define SYCL_ELSE_IF(condition)	else if(condition)

#define SYCL_WHILE(condition) while(condition)

#define SYCL_FOR(init, condition, increment) for(init; condition; increment)

#endif // SYCL_GTX
