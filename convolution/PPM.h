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

	PPM(string filename) {
		using namespace std;

		vector<string> lines;
		int size;
		{
			auto file = ifstream(filename);
			string line;
			getline(file, line); // 1
			getline(file, line); // 2
			istringstream stream(line);
			stream >> width >> height;
			getline(file, line); // 3

			size = width * height;
			lines.resize(size);
			data.reserve(size);

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

			data.emplace_back(r, g, b);
		}
	}

	PPM(const vector<float>& internal, int width, int height)
		: width(width), height(height)
	{
		int size = internal.size();
		data.reserve((size / 4) * 3);

		for(int i = 0; i < size; i += 4) {
			data.emplace_back(internal[i], internal[i + 1], internal[i + 2]);
		}
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

	vector<float> toInternal() const {
		vector<float> internal;
		internal.reserve(data.size() * 4);
		for(auto& p : data) {
			internal.push_back(p.r);
			internal.push_back(p.g);
			internal.push_back(p.b);
			internal.push_back(0);
		}
		return internal;
	}
};
