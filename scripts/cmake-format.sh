#!/bin/sh

repoRoot=$1
cmakeFormatExe=cmake-format

find $repoRoot -regex ".*\(CMakeLists\.txt\|\.cmake\)" \
	       -exec $cmakeFormatExe -i {} \;
