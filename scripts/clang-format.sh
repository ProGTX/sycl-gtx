#!/bin/sh

repoRoot=$1
clangFormatExe=clang-format-4.0

find $repoRoot -regex ".*\.\(h\|cpp\)" -exec $clangFormatExe -i {} \;
