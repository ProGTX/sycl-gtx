#include "PPM.h"

#include <sstream>
#include <streambuf>

using namespace std;


void PPM::P3(ifstream& file, int size) {
	vector<string> lines;
	lines.resize(size);

	for(int i = 0; i < size; ++i) {
		getline(file, lines[i]);
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

void PPM::P6(string filename, int size, int start) {
	size *= 3;
	vector<char> input;
	input.resize(size);

	// http://codereview.stackexchange.com/a/22907
	auto file = ifstream(filename, ios::binary);
	file.seekg(start, ios::beg);
	file.read(input.data(), size);

	int r, g, b;
	for(int i = 0; i < size; i += 3) {
		r = (int)input[i];
		g = (int)input[i + 1];
		b = (int)input[i + 2];
		data.emplace_back(r, g, b);
	}
}

PPM::PPM(string filename) {
	auto file = ifstream(filename);
	string line;
	int offset = 0;

	auto tryGetLine = [&]() {
		do {
			getline(file, line);
			offset += line.length() + 1; // +1 for the eaten newline
		}
		while(line[0] == '#');
	};

	tryGetLine(); // Format
	isBinary = line.find("P6") != string::npos;

	tryGetLine(); // Width and height
	istringstream stream(line);
	stream >> width >> height;

	tryGetLine(); // Levels, ignore

	auto size = width * height;
	data.reserve(size);

	if(isBinary) {
		P6(filename, size, offset);
	}
	else {
		P3(file, size);
	}
}

PPM::PPM(const vector<float>& internal, int width, int height)
	: width(width), height(height) {
	int size = internal.size();
	data.reserve((size / 4) * 3);

	for(int i = 0; i < size; i += 4) {
		data.emplace_back(internal[i], internal[i + 1], internal[i + 2]);
	}
}

void PPM::store(string filename) const {
	FILE* file = fopen(filename.c_str(), "w");
	fprintf(file, "P3\n%d %d\n%d\n", width, height, 255);
	for(int i = 0; i < width*height; ++i) {
		auto& p = data[i];
		fprintf(file, "%d %d %d\n", Pixel::toInt(p.r), Pixel::toInt(p.g), Pixel::toInt(p.b));
	}
	fclose(file);
}

vector<float> PPM::toInternal() const {
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
