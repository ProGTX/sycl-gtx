#pragma once

namespace cl {
namespace sycl {
namespace detail {

template <size_t dimensions>
struct point {
private:
public:
	size_t get(int dimension) const;
	size_t& operator[](int dimension);

	point& operator=(const point& rhs);
	point& operator+=(const point& rhs);
	point& operator*=(const point& rhs);
	point& operator/=(const point& rhs);
	point& operator%=(const point& rhs);
	point& operator>>=(const point& rhs);
	point& operator<<=(const point& rhs);
	point& operator&=(const point& rhs);
	point& operator^=(const point& rhs);
	point& operator|=(const point& rhs);
};

} // namespace detail
} // namespace sycl
} // namespace cl
