#!/bin/sh

repoRoot=$1
clangFormatExe=clang-format-5.0

find $repoRoot -regex ".*\.\(h\|cpp\)" -exec $clangFormatExe -i {} \;
