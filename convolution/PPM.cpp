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
	int size = width * height;
	vector<char> output;

	// Estimation for non binary: three colors, each color at most three digits plus a space
	static const int maxLine = 3 * 4;

	{
		stringstream headerStream;
		headerStream << 'P' << (isBinary ? 6 : 3) << '\n' << width << ' ' << height << "\n255\n";
		auto header = headerStream.str();
		output.insert(output.end(), header.begin(), header.end());
	}
	int offset = output.size();

	if(isBinary) {
		output.reserve(size * 3 + offset);
	}
	else {
		output.reserve(size * maxLine + offset);
	}

	char temp[maxLine + 1]; // +1 for zero termination
	auto tempLength = [&temp]() {
		int i = 0;
		while(temp[i] != 0) {
			++i;
		}
		return i;
	};

	int r, g, b;
	for(int i = 0; i < size; ++i) {
		auto& p = data[i];

		r = Pixel::toInt(p.r);
		g = Pixel::toInt(p.g);
		b = Pixel::toInt(p.b);

		if(isBinary) {
			output.push_back(r);
			output.push_back(g);
			output.push_back(b);
		}
		else {
			sprintf(temp, "%d %d %d\n", r, g, b);
			output.insert(output.end(), temp, temp + tempLength());
		}
	}

	auto file = ofstream(filename, ios::binary);
	file.write(output.data(), output.size());
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
