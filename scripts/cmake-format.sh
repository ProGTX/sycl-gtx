#!/bin/sh

repoRoot="${1:-.}"
cmakeFormatExe="${2:-cmake-format}"

find $repoRoot -regex ".*\(CMakeLists\.txt\|\.cmake\)" \
	       -exec $cmakeFormatExe -i {} \;
