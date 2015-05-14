#include "tests.h"

// Parallel reduction sum with local decomposition

bool test6() {
	using namespace cl::sycl;

	{
		queue myQueue;

		static const auto group_size = myQueue.get_device().get_info<CL_DEVICE_MAX_WORK_GROUP_SIZE>();
		static const auto size = group_size * 16;

		buffer<float> ping(size);
		buffer<float> pong(size);

		auto P = &ping;
		auto Q = &pong;

		// Init
		command_group(myQueue, [&]() {
			auto p = P->get_access<access::write>();

			parallel_for<>(range<1>(size), [=](id<1> index) {
				p[index] = index;
			});
		});

		for(unsigned int N = size / 2; N > 0; N /= 2 * group_size) {
			/*
			clErr = CL_SUCCESS;
			clErr |= clSetKernelArg(Kernel, 0, sizeof(cl_mem), &m_dPingArray);
			clErr |= clSetKernelArg(Kernel, 1, sizeof(cl_mem), &m_dPongArray);
			clErr |= clSetKernelArg(Kernel, 2, sizeof(cl_uint), &N);
			V_RETURN_CL(clErr, "Error setting kernel parameters.");

			// Allocate shared (local) memory for the kernel
			clErr = clSetKernelArg(Kernel, 3, LocalWorkSize[0] * sizeof(cl_uint), nullptr);
			V_RETURN_CL(clErr, "Error allocating shared memory.");

			globalWorkSize[0] = CLUtil::GetGlobalWorkSize(N, LocalWorkSize[0]);
			clErr = clEnqueueNDRangeKernel(
				CommandQueue, Kernel, 1, nullptr, globalWorkSize, LocalWorkSize, 0, nullptr, nullptr
				);
			V_RETURN_CL(clErr, "Error enqueuing decomposition kernel.");
			*/

			command_group(myQueue, [&]() {
				auto p = P->get_access<access::read_write>();
				auto q = Q->get_access<access::read_write>();

				parallel_for<>(nd_range<1>(size, group_size), [=](nd_item<1> index) {
					/*
					int gid = get_global_id(0);
					int lid = get_local_id(0);

					int second = gid + N;
					if(second < 2 * N) {
						localBlock[lid] = inArray[gid] + inArray[second];
					}
					barrier(CLK_LOCAL_MEM_FENCE);

					N = min(N, get_local_size(0));

					for(uint stride = N / 2; stride > 0; stride /= 2) {
						if(lid < stride) {
							localBlock[lid] += localBlock[lid + stride];
						}
						barrier(CLK_LOCAL_MEM_FENCE);
					}

					if(lid == 0) {
						outArray[gid / N] = localBlock[0];
					}
					*/
				});
			});

			std::swap(P, Q);
		}

		auto p = P->get_access<access::read, access::host_buffer>();
		int sum = ((uint64_t)size * (uint64_t)(size - 1)) / 2;

		if(p[0] != sum) {
			debug() << "wrong sum, should be" << sum << "- is" << p[0];
			return false;
		}
	}

	return true;
}
