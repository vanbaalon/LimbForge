#!/bin/sh
set -eu
# Run from the repository root, after building wolfnum_exact_gemm4.
clang++ -std=c++17 -O2 -I include -I tests -I benchmarks -I benchmarks/experiments/exact_gemm4 \
  -I /opt/homebrew/opt/mpfr/include -I /opt/homebrew/opt/gmp/include \
  benchmarks/experiments/exact_gemm4/comparison.cpp build/libwolfnum_exact_gemm4.a build/liblimbforge.a \
  -L /opt/homebrew/opt/mpfr/lib -L /opt/homebrew/opt/gmp/lib -lmpfr -lgmp \
  -framework Metal -framework Foundation -o build/wolfnum_exact_gemm4_comparison
