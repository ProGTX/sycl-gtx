// Khairi Reda
// evl.uic.edu/kreda/gpu/image-convolution/
// Taken from 3_reduced.cl

__kernel void convolute(
	const __global float4* input,
	__global float4* output,
	__constant float4* filter __attribute__((max_constant_size(4096)))
) {
	int3 gid = (int3)(get_global_id(0), get_global_id(1), 0);
	
	if(
		gid.x < HALF_FILTER_SIZE								||
		gid.x > IMAGE_W - HALF_FILTER_SIZE - 1	||
		gid.y < HALF_FILTER_SIZE								||
		gid.y > IMAGE_H - HALF_FILTER_SIZE - 1
	) {
		// Not performed at the edges, but not much of a problem
		return;
	}

	gid.z = gid.y * IMAGE_W + gid.x;

	int filterPos = 0;
	float4 sum = (float4) 0.0;
	for(int r = -HALF_FILTER_SIZE; r <= HALF_FILTER_SIZE; ++r) {
		int inputPos = gid.z + IMAGE_W * r;
		for(int c = -HALF_FILTER_SIZE; c <= HALF_FILTER_SIZE; ++c) {
			sum += input[inputPos] * filter[filterPos];
			filterPos += 1;
			inputPos += 1;
		}
	}
	output[gid.z] = sum;
}
