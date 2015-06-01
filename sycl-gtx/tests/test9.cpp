#include "tests.h"

// Work efficient prefix sum

template <typename F, typename std::enable_if<std::is_floating_point<F>::value>::type* = nullptr>
bool check_sum(cl::sycl::buffer<F, 1>& data) {
	using namespace cl::sycl;

	auto d = data.get_access<access::read, access::host_buffer>();
	F sum = 0;
	for(size_t i = 0; i < data.get_count(); ++i) {
		sum += (F)i;
		auto diff = std::abs(sum - d[i]);
		if(diff > 0.01) {
			debug() << "wrong sum, should be" << sum << "- is" << d[i];
			return false;
		}
	}

	return true;
}

template <typename I, typename std::enable_if<std::is_integral<I>::value>::type* = nullptr>
bool check_sum(cl::sycl::buffer<I, 1>& data) {
	using namespace cl::sycl;

	auto d = data.get_access<access::read, access::host_buffer>();
	I sum = 0;
	for(size_t i = 0; i < data.get_count(); ++i) {
		sum += (I)i;
		if(d[i] != sum) {
			debug() << "wrong sum, should be" << sum << "- is" << d[i];
			return false;
		}
	}

	return true;
}

template <typename T>
struct prefix_sum_kernel {
protected:
	using mode = cl::sycl::access::mode;
	using target = cl::sycl::access::target;
	using buffer = cl::sycl::buffer<T>;

	// Global memory
	cl::sycl::accessor<T> input;
	cl::sycl::accessor<T, 1, mode::write> higher_level;

	// Local memory
	cl::sycl::accessor<T, 1, mode::read_write, target::local> localBlock;

public:
	prefix_sum_kernel(cl::sycl::handler& cgh, buffer& data, buffer& higher_level, size_t group_size)
		: input(data.get_access<mode::read_write>(cgh)),
		higher_level(higher_level.get_access<mode::write>(cgh)),
		localBlock(group_size) {}

	void operator()(cl::sycl::nd_item<1> index) {
		using namespace cl::sycl;

		uint1 GID = 2 * index.get_global(0);
		uint1 LID = 2 * index.get_local(0);

		localBlock[LID] = input[GID];
		localBlock[LID + 1] = input[GID + 1];

		index.barrier(access::fence_space::local);

		uint1 N = 2 * index.get_local(0);
		uint1 first;
		uint1 second;

		LID /= 2;

		// Up-sweep
		uint1 offset = 1;
		SYCL_WHILE(offset < N)
			SYCL_BEGIN {
			SYCL_IF(LID % offset == 0)
			SYCL_BEGIN {
				first = 2 * LID + offset - 1;
				second = first + offset;
				localBlock[second] = localBlock[first] + localBlock[second];
			}
			SYCL_END

			index.barrier(access::fence_space::local);
			offset *= 2;
		}
		SYCL_END

		SYCL_IF(LID == 0)
		SYCL_THEN({
			localBlock[N - 1] = 0;
		})
		index.barrier(access::fence_space::local);

		vec<T, 1> tmp;

		// Down-sweep
		offset = N;
		SYCL_WHILE(offset > 0)
		SYCL_BEGIN {
			SYCL_IF(LID % offset == 0)
			SYCL_BEGIN {
				first = 2 * LID + offset - 1;
				second = first + offset;
				tmp = localBlock[second];
				localBlock[second] = localBlock[first] + tmp;
				localBlock[first] = tmp;
			}
			SYCL_END

			index.barrier(access::fence_space::local);
			offset /= 2;
		}
		SYCL_END

		LID *= 2;
		input[GID] += localBlock[LID];
		input[GID + 1] += localBlock[LID + 1];
	}
};

template <typename T>
struct prefix_sum_join_kernel {
protected:
	using mode = cl::sycl::access::mode;
	using target = cl::sycl::access::target;
	using buffer = cl::sycl::buffer<T>;

	// Global memory
	cl::sycl::accessor<T> data;
	cl::sycl::accessor<T, 1, mode::read> higher_level;

public:
	prefix_sum_join_kernel(cl::sycl::handler& cgh, buffer& data, buffer& higher_level)
		: data(data.get_access<mode::read_write>(cgh)),
		higher_level(higher_level.get_access<mode::read>(cgh)) {}

	void operator()(cl::sycl::nd_item<1> index) {
		using namespace cl::sycl;
		int1 GID = index.get_global(0);
		int1 N = 2 * index.get_local(0);
		SYCL_IF(GID >= N)
		SYCL_THEN({
			data[GID] += higher_level[GID / N - 1];
		})
	}
};

bool test9() {
	using namespace cl::sycl;

	{
		queue myQueue;

		const auto group_size = myQueue.get_device().get_info<CL_DEVICE_MAX_WORK_GROUP_SIZE>();
		const auto size = group_size * 2;
		size_t N = size;
		using type = double;

		// Calculate required number of buffer levels
		N = size;
		size_t number_levels = 1;
		while(N > 0) {
			N /= 2 * group_size;
			++number_levels;
		}

		// Create buffers
		vector_class<buffer<type>> data;
		data.reserve(number_levels);
		N = size;
		for(size_t i = 0; i < number_levels; ++i) {
			data.emplace_back(N);
			N = std::max(N / (2 * group_size), group_size);
		}

		// Init
		myQueue.submit([&](handler& cgh) {
			auto d = data[0].get_access<access::write>();

			parallel_for<>(range<1>(size), [=](id<1> index) {
				d[index] = index;
			cgh.parallel_for<>(range<1>(size), [=](id<1> index) {
			});
		});

		myQueue.submit([&](handler& cgh) {
			N = size;

			cgh.parallel_for<>(
				nd_range<1>(N / 2, group_size),
				prefix_sum_kernel<type>(cgh, data[0], data[1], 2*group_size)
			);

			if(N <= group_size * 2) {
				return;
			}

			// TODO: Extend for large arrays
		});

		return check_sum(data[0]);
	}

	return true;
}
