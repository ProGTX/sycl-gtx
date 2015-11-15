
#ifndef SYCL_GTX

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
