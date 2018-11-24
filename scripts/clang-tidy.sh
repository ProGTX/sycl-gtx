#!/bin/sh

repoRoot="${1:-.}"
clangTidyExe="${2:-clang-tidy-6.0}"
flags="${3:--fix}"

find "$repoRoot" -name "*.cpp" \
     -exec "$clangTidyExe" -header-filter=.* $flags {} \;
