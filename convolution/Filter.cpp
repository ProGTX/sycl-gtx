#include "Filter.h"

// Based on: Khairi Reda
// evl.uic.edu/kreda/gpu/image-convolution/

#include <cmath>


using std::vector;

const vector<float> Filter::gaussian3 = {
	1.f / 16.f,	2.f / 16.f,	1.f / 16.f,
	2.f / 16.f,	4.f / 16.f,	2.f / 16.f,
	1.f / 16.f,	2.f / 16.f,	1.f / 16.f
};

const vector<float> Filter::gaussian5 = {
	1.f / 273.f,	4.f / 273.f,	7.f / 273.f,	4.f / 273.f,	1.f / 273.f,
	4.f / 273.f,	16.f / 273.f,	26.f / 273.f,	16.f / 273.f,	4.f / 273.f,
	7.f / 273.f,	26.f / 273.f,	41.f / 273.f,	26.f / 273.f,	7.f / 273.f,
	4.f / 273.f,	16.f / 273.f,	26.f / 273.f,	16.f / 273.f,	4.f / 273.f,
	1.f / 273.f,	4.f / 273.f,	7.f / 273.f,	4.f / 273.f,	1.f / 273.f,
};

vector<float> Filter::generate(int size) {
	static const float PI = 3.14159265358979323846f;
	const float DELTA = 1.84089642f * ((float)size / 7.f);
	const float TWO_DELTA_SQ = 2.0f * DELTA * DELTA;
	const float k = 1.0f / (PI * TWO_DELTA_SQ);

	vector<float> filter(size * size * 4);
	int w = size / 2;

	const vector<float>* precomputed = nullptr;

	if(size == 3) {
		precomputed = &gaussian3;
	}
	else if(size == 5) {
		precomputed = &gaussian5;
	}

	int i = 0;
	for(int r = -w; r <= w; ++r) {
		for(int c = -w; c <= w; ++c) {
			float value;
			if(precomputed != nullptr) {
				value = precomputed->operator[](i / 4);
			}
			else {
				value = k * exp(-(r*r + c*c) / TWO_DELTA_SQ);
			}

			filter[i++] = value;
			filter[i++] = value;
			filter[i++] = value;
			filter[i++] = r == c && c == 0 ? 1.0f : 0.0f;
		}
	}

	return filter;
}
