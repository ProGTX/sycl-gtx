#pragma once

#include <chrono>

struct Timer {
public:
	using time_point = decltype(std::chrono::high_resolution_clock::now());

private:
	time_point start;

public:
	static time_point now() {
		return std::chrono::high_resolution_clock::now();
	};

	Timer()
		: start(now()) {}

	void reset() {
		start = now();
	}

	time_point startPoint() const {
		return start;
	}

	long long elapsed() const {
		return std::chrono::duration_cast<std::chrono::microseconds>(now() - start).count();
	}
};
