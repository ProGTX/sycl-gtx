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

	static PPM load(string filename) {
		using namespace std;
		auto image = PPM();

		vector<string> lines;
		int size;
		{
			auto file = ifstream(filename);
			string line;
			getline(file, line); // 1
			getline(file, line); // 2
			istringstream stream(line);
			stream >> image.width >> image.height;
			getline(file, line); // 3
			size = image.width*image.height;
			lines.resize(size);
			image.data.reserve(size);

			for(int i = 0; i < size; ++i) {
				getline(file, lines[i]);
			}
		}

		size_t previous;
		size_t current;
		auto fetch = [&previous, &current](const string& line) {
			previous = current + 1;
			current = line.find(' ', previous);
			return atoi(line.substr(previous, current - previous).c_str()); 
		};

		int r, g, b;
		for(int i = 0; i < size; ++i) {
			auto& line = lines[i];
			current = -1;

			r = fetch(line);
			g = fetch(line);
			b = fetch(line);

			image.data.emplace_back(r, g, b);
		}

		return image;
	}

	void store(string filename) const {
		using namespace std;

		FILE* file = fopen(filename.c_str(), "w");
		fprintf(file, "P3\n%d %d\n%d\n", width, height, 255);
		for(int i = 0; i < width*height; ++i) {
			auto& p = data[i];
			fprintf(file, "%d %d %d\n", Pixel::toInt(p.r), Pixel::toInt(p.g), Pixel::toInt(p.b));
		}
		fclose(file);
	}
};
