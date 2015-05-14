#include "tests.h"

// Parallel reduction sum with local decomposition

bool test6() {
	using namespace cl::sycl;

	{
		queue myQueue;

		const auto group_size = myQueue.get_device().get_info<CL_DEVICE_MAX_WORK_GROUP_SIZE>();
		const auto size = group_size * 16;

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

				parallel_for<>(nd_range<1>(N, group_size), [=](nd_item<1> index) {
					auto gid = index.get_global_id(0);
					auto lid = index.get_local_id(0);
					auto N = index.get_global_range()[0];
					auto second = gid + N;

					SYCL_IF(second < 2 * N)
					SYCL_THEN({
						// localBlock[lid] = p[gid] + p[second];
					})

					index.barrier(access::fence_space::local);

					// N = min(N, index.get_local_range()[0]);

					SYCL_FOR(size_t stride = N / 2, stride > 0, stride /= 2)
					SYCL_BLOCK({
						SYCL_IF(lid < stride)
						SYCL_THEN({
							// localBlock[lid] += localBlock[lid + stride];
						})
						index.barrier(access::fence_space::local);
					})

					SYCL_IF(lid == 0)
					SYCL_THEN({
						// q[gid / N] = localBlock[0];
					})
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
