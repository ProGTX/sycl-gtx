#pragma once

#include <vector>


struct Filter {
	static const std::vector<float> gaussian3;
	static const std::vector<float> gaussian5;

	static std::vector<float> generate(int size);
};
