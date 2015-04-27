#pragma once

#include "specification\access.h"

#include <iostream>
#include <sstream>

// Visual Studio 2013 still lacks some support for modern C++
#if _MSC_VER <= 1800
#define MSVC_LOW 1
#endif

#if MSVC_LOW
#define __func__ __FUNCTION__
#define constexpr const
#endif

#define DSELF() debug(__func__)

class debug {
protected:
	static constexpr bool DEBUG_ACTIVE = true;
	std::stringstream stream;
	std::stringstream before;

	void AddToStream(cl::sycl::access::mode mode) {
		if(DEBUG_ACTIVE) {
			using namespace cl::sycl::access;
			stream << "mode::";
			switch(mode) {
				case read:
					stream << "read ";
					break;
				case write:
					stream << "write ";
					break;
				case atomic:
					stream << "atomic ";
					break;
				case read_write:
					stream << "read_write ";
					break;
				case discard_read_write:
					stream << "discard_read_write ";
					break;
			}
		}
	}

	void AddToStream(cl::sycl::access::target target) {
		if(DEBUG_ACTIVE) {
			using namespace cl::sycl::access;
			stream << "target::";
			switch(target) {
				case global_buffer:
					stream << "global_buffer ";
					break;
				case constant_buffer:
					stream << "constant_buffer ";
					break;
				case local:
					stream << "local ";
					break;
				case image:
					stream << "image ";
					break;
				case host_buffer:
					stream << "host_buffer ";
					break;
				case host_image:
					stream << "host_image ";
					break;
				case image_array:
					stream << "image_array ";
					break;
				case cl_buffer:
					stream << "cl_buffer ";
					break;
				case cl_image:
					stream << "cl_image ";
					break;
			}
		}
	}

	template<typename T>
	void AddToStream(T add) {
		if(DEBUG_ACTIVE) {
			stream << add << ' ';
		}
	}

	template<class T>
	void AddToStream(std::basic_string<T> string) {
		if(DEBUG_ACTIVE) {
			stream << string << ' ';
		}
	}

	template<template<class, class...> class Container, class First, class... Others>
	void AddToStream(Container<First, Others...> list) {
		for(auto&& element : list) {
			AddToStream(element);
		}
	}

public:
	debug() = default;
#if MSVC_LOW
	debug(debug&& move)
		: stream(std::move(move.stream)), before(std::move(move.before)) {}
#elif
	debug(debug&& move) = default;
#endif
	debug(const debug& copy) = delete;

	template<typename T>
	debug(T add)
		: debug() {
		before << "Debug: ";
		AddToStream(add);
	}

	template<typename T>
	debug& operator<<(T add) {
		AddToStream(add);
		return *this;
	}

	template<typename U, typename T>
	debug(U before, T add)
		: debug() {
		this->before << before;
		AddToStream(add);
	}

	virtual ~debug() {
		if(DEBUG_ACTIVE) {
			std::cout << before.str() << stream.str() << std::endl;
		}
	}

	template<typename T>
	static debug warning(T message) {
		return debug("SYCL warning: ", message);
	}
};
