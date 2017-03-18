#!/bin/sh

repoRoot=$1
clangTidyExe=clang-tidy-4.0

find $repoRoot -name "*.cpp" -exec $clangTidyExe -header-filter=.* -fix {} \;
