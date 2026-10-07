# Related work

GPU multiprecision already exists. [CAMPARY's arithmetic paper](https://perso.ens-lyon.fr/jean-michel.muller/07118139.pdf)
implements extended precision through floating-point expansions on CPU and GPU.
[CUMP](https://github.com/skystar0227/CUMP) provides GMP-like CUDA floating-point
arithmetic, but its repository is archived. [MPRES-BLAS](https://github.com/kisupov/mpres-blas)
provides CUDA multiprecision linear algebra and basic arithmetic; its README
notes MPFR dependencies for some operations, including division.
[NVIDIA CGBN](https://github.com/NVlabs/CGBN) covers fixed-width unsigned integers,
which alone do not supply the solver's real/complex arithmetic.

A newer [mpc_cuda project](https://github.com/tkouya/mpc_cuda/blob/main/README.md)
advertises CUDA ports of GMP/MPFR/MPC plus fixed-precision fast paths. That is a
useful candidate for an NVIDIA implementation; its performance and correctness
claims were not independently tested here. These CUDA implementations do not
directly execute on this Mac's Metal GPU. LimbForge addresses that local backend
need; it does not introduce GPU multiprecision as a new capability.
