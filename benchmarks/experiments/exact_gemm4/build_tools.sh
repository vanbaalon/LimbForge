#!/bin/sh
set -eu
sh benchmarks/experiments/exact_gemm4/build_comparison.sh
for task in supplement clock_probe; do
  target="test_wolfnum_exact_gemm4_supplement"
  if [ "$task" = clock_probe ]; then target="wolfnum_exact_gemm4_clock_probe"; fi
  clang++ -std=c++17 -O2 -I include -I tests -I benchmarks/experiments/exact_gemm4 \
    -I /opt/homebrew/opt/mpfr/include -I /opt/homebrew/opt/gmp/include \
    "benchmarks/experiments/exact_gemm4/$task.cpp" build/libwolfnum_exact_gemm4.a build/liblimbforge.a \
    -L /opt/homebrew/opt/mpfr/lib -L /opt/homebrew/opt/gmp/lib -lmpfr -lgmp \
    -framework Metal -framework Foundation -o "build/$target"
done
