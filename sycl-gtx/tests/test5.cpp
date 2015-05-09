#include "tests.h"
#include <ctime>

// Passing functors as nd-range kernels

using namespace cl::sycl;

class example_functor {
public:
	using rw_acc_t = accessor<int, 1, access::read_write, access::global_buffer>;

private:
	rw_acc_t ptr;
	int random_num;

public:
	example_functor(rw_acc_t p)
		: ptr(p) {
		random_num = std::rand() % (100 - 1) + 1;
	}

	void operator()(item<1> item) {
		ptr[item.get_global_id()] = random_num;
	}

	int get_random() {
		return random_num;
	}
};

bool test5() {
	int data[64];
	srand(time(0));

	{
		queue myQueue;

		buffer<int, 1> buf(data, range<1>(64));

		command_group(myQueue, [&]() {
			auto ptr = buf.get_access<access::read_write>();

			parallel_for(nd_range<1>(range<1>(64), range<1>(8)),
				example_functor(ptr)
			);
		});

		{
			int random_num = 0;

			command_group(myQueue, [&]() {
				auto ptr = buf.get_access<access::read_write>();

				auto functor = example_functor(ptr);

				parallel_for(nd_range<1>(range<1>(64), range<1>(8)),
					functor
				);

				random_num = functor.get_random();

				debug() << "-> Random number " << random_num;

			});

			auto hostPtr = buf.get_access<access::read_write, access::host_buffer>();

			if(hostPtr[5] != random_num) {
				debug() << "The data retrieved from the device " << hostPtr[5]
					<< "does not match the random number generated: "
					<< random_num;
				return false;
			}
		}
	}

	return true;
}
