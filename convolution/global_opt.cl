// Khairi Reda
// evl.uic.edu/kreda/gpu/image-convolution/
// Taken from 3_reduced.cl

__kernel void convolute(
	const __global float4 * input, 
	__global float4 * output,
	__constant float4 * filter __attribute__((max_constant_size(4096)))
)
{
	int rowOffset = get_global_id(1) * IMAGE_W;
	int my = get_global_id(0) + rowOffset;

	if (
		get_global_id(0) < HALF_FILTER_SIZE					||
		get_global_id(0) > IMAGE_W - HALF_FILTER_SIZE - 1	|| 
		get_global_id(1) < HALF_FILTER_SIZE					||
		get_global_id(1) > IMAGE_H - HALF_FILTER_SIZE - 1
	)
	{
		return;
	}
	
	else
	{
		// perform convolution
		int fIndex = 0;
		float4 sum = (float4) 0.0;
		
		for (int r = -HALF_FILTER_SIZE; r <= HALF_FILTER_SIZE; r++)
		{
			int curRow = my + r * IMAGE_W;
			for (int c = -HALF_FILTER_SIZE; c <= HALF_FILTER_SIZE; c++)
			{	
				sum += input[ curRow + c ] * filter[ fIndex ]; 
				fIndex++;
	
			}
		}
		output[my] = sum;
	}
}
