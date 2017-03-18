#include "SYCL/ranges/point.h"

using namespace cl::sycl;
using namespace detail;

const string_class point_names::id_global = "_sycl_gid";
const string_class point_names::range_global = "_sycl_grange";
const string_class point_names::id_local = "_sycl_lid";
const string_class point_names::range_local = "_sycl_lrange";
