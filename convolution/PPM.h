#pragma once

#include <string>
#include <vector>


struct Pixel {
	static const int levels = 255;
	float r = 0;
	float g = 0;
	float b = 0;

	Pixel(int r, int g, int b)
		: r((float)r / levels), g((float)g / levels), b((float)b / levels) {}

	Pixel(float r, float g, float b)
		: r(r), g(g), b(b) {}

	static int toInt(float value) {
		return (int)(value * levels);
	}
};

struct PPM {
	using string = std::string;
	template <class T>
	using vector = std::vector<T>;

	int width;
	int height;
	vector<Pixel> data;


	PPM() {}
	PPM(string filename);
	PPM(const vector<float>& internal, int width, int height);
	void store(string filename) const;
	vector<float> toInternal() const;
};
