#pragma once

#include <fstream>
#include <sstream>
#include <string>
#include <vector>

struct Pixel {
	static const int levels = 255;
	float r = 0;
	float g = 0;
	float b = 0;

	Pixel(int r, int g, int b)
		: r((float)r / levels), g((float)g / levels), b((float)b / levels) {}

	int operator[](int i) {
		float* c = nullptr;
		if(i == 0) {
			c = &r;
		}
		else if(i == 1) {
			c = &g;
		}
		else if(i == 2) {
			c = &b;
		}
		return (int)(*c * (float)levels);
	}
};

struct PPM {
	using string = std::string;
	template <class T>
	using vector = std::vector<T>;

	int width;
	int height;
	vector<Pixel> data;

	static PPM load(string filename) {
		using namespace std;
		auto file = ifstream(filename);
		auto image = PPM();

		string tmpStr;
		file >> tmpStr >> image.width >> image.height >> tmpStr;
		auto size = image.width*image.height;
		image.data.reserve(size);

		int r, g, b;

		for(int i = 0; i < size; ++i) {
			file >> r >> g >> b;
			image.data.emplace_back(r, g, b);
		}

		return image;
	}

	void store(string filename) {
		using namespace std;

		stringstream file;
		file << "P3" << endl << width << ' ' << height << endl << Pixel::levels << endl;

		for(int i = 0; i < width*height; ++i) {
			file << data[i][0] << ' ' << data[i][1] << ' ' << data[i][2] << endl;
		}
		ofstream(filename) << file.str();
	}
};
