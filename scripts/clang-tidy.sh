#!/bin/sh

repoRoot=${1:-.}
clangTidyExe=${2:-clang-tidy-5.0}

find $repoRoot -name "*.cpp" -exec $clangTidyExe -header-filter=.* -fix {} \;
