#include "section9_p13_gemm_break_even_measured.hpp"
#include <iostream>
int main(){auto table=section9_gemm_break_even();unsigned checked=0;
{limbforge::DispatchKey key{"complex_gemm/complex_composed",352,1,66,3800,130,18,false,"WolfNum 1.3.1 / Apple M5 Max / Release / busy-host library; no idle consumer acceptance; measured=Apple M5 Max / 1.3.1 / complex-gemm-contracts; binary=a8a40fbbf9a3290cd200d4c24d0d632554c6d739b168c55cf2892a9e15725d11; archive=c9dd6ebed304b6f74258f8ea0aebb4c29e1b205fff4deb115254ee5652ff9d24; clock=cpu-interleaved; cpu=MPFR/complex_composed; active_workers=18; load_series=e71b047ca3ee9e662fc7cd733275671272b959d71e4f5e5a4a68b07ea8fd93c8; all_loadavg_below_4=false; scope=numeric-array-library; bridge=excluded",false};
if(table.recommend(key)!=limbforge::BackendRecommendation::gpu)return 1;
auto unknown=key;unknown.bits+=32;if(table.recommend(unknown)!=limbforge::BackendRecommendation::unknown)return 2;
unknown=key;unknown.profile+="/unmeasured";if(table.recommend(unknown)!=limbforge::BackendRecommendation::unknown)return 3;
unknown=key;unknown.k+=1;if(table.recommend(unknown)!=limbforge::BackendRecommendation::unknown)return 4;
unknown=key;unknown.resident=!key.resident;unknown.profile+="/unmeasured-storage";if(table.recommend(unknown)!=limbforge::BackendRecommendation::unknown)return 5;
++checked;}
{limbforge::DispatchKey key{"complex_gemm/complex_fused",352,1,66,3800,130,18,true,"WolfNum 1.3.1 / Apple M5 Max / Release / busy-host library; no idle consumer acceptance; measured=Apple M5 Max / 1.3.1 / complex-gemm-contracts; binary=a8a40fbbf9a3290cd200d4c24d0d632554c6d739b168c55cf2892a9e15725d11; archive=c9dd6ebed304b6f74258f8ea0aebb4c29e1b205fff4deb115254ee5652ff9d24; clock=cpu-interleaved; cpu=MPFR/complex_fused; active_workers=18; load_series=e71b047ca3ee9e662fc7cd733275671272b959d71e4f5e5a4a68b07ea8fd93c8; all_loadavg_below_4=false; scope=numeric-array-library; bridge=excluded",false};
if(table.recommend(key)!=limbforge::BackendRecommendation::gpu)return 1;
auto unknown=key;unknown.bits+=32;if(table.recommend(unknown)!=limbforge::BackendRecommendation::unknown)return 2;
unknown=key;unknown.profile+="/unmeasured";if(table.recommend(unknown)!=limbforge::BackendRecommendation::unknown)return 3;
unknown=key;unknown.k+=1;if(table.recommend(unknown)!=limbforge::BackendRecommendation::unknown)return 4;
unknown=key;unknown.resident=!key.resident;unknown.profile+="/unmeasured-storage";if(table.recommend(unknown)!=limbforge::BackendRecommendation::unknown)return 5;
++checked;}
{limbforge::DispatchKey key{"complex_gemm/exact_real_embedding",352,1,66,3800,130,18,false,"WolfNum 1.3.1 / Apple M5 Max / Release / busy-host library; no idle consumer acceptance; measured=Apple M5 Max / 1.3.1 / complex-gemm-contracts; binary=a8a40fbbf9a3290cd200d4c24d0d632554c6d739b168c55cf2892a9e15725d11; archive=c9dd6ebed304b6f74258f8ea0aebb4c29e1b205fff4deb115254ee5652ff9d24; clock=cpu-interleaved; cpu=MPFR/exact_real_embedding; active_workers=18; load_series=e71b047ca3ee9e662fc7cd733275671272b959d71e4f5e5a4a68b07ea8fd93c8; all_loadavg_below_4=false; scope=numeric-array-library; bridge=excluded",false};
if(table.recommend(key)!=limbforge::BackendRecommendation::gpu)return 1;
auto unknown=key;unknown.bits+=32;if(table.recommend(unknown)!=limbforge::BackendRecommendation::unknown)return 2;
unknown=key;unknown.profile+="/unmeasured";if(table.recommend(unknown)!=limbforge::BackendRecommendation::unknown)return 3;
unknown=key;unknown.k+=1;if(table.recommend(unknown)!=limbforge::BackendRecommendation::unknown)return 4;
unknown=key;unknown.resident=!key.resident;unknown.profile+="/unmeasured-storage";if(table.recommend(unknown)!=limbforge::BackendRecommendation::unknown)return 5;
++checked;}
{limbforge::DispatchKey key{"complex_gemm/gauss_three_real_composed",352,1,66,3800,130,18,false,"WolfNum 1.3.1 / Apple M5 Max / Release / busy-host library; no idle consumer acceptance; measured=Apple M5 Max / 1.3.1 / complex-gemm-contracts; binary=a8a40fbbf9a3290cd200d4c24d0d632554c6d739b168c55cf2892a9e15725d11; archive=c9dd6ebed304b6f74258f8ea0aebb4c29e1b205fff4deb115254ee5652ff9d24; clock=cpu-interleaved; cpu=MPFR/gauss_three_real_composed; active_workers=18; load_series=e71b047ca3ee9e662fc7cd733275671272b959d71e4f5e5a4a68b07ea8fd93c8; all_loadavg_below_4=false; scope=numeric-array-library; bridge=excluded",false};
if(table.recommend(key)!=limbforge::BackendRecommendation::gpu)return 1;
auto unknown=key;unknown.bits+=32;if(table.recommend(unknown)!=limbforge::BackendRecommendation::unknown)return 2;
unknown=key;unknown.profile+="/unmeasured";if(table.recommend(unknown)!=limbforge::BackendRecommendation::unknown)return 3;
unknown=key;unknown.k+=1;if(table.recommend(unknown)!=limbforge::BackendRecommendation::unknown)return 4;
unknown=key;unknown.resident=!key.resident;unknown.profile+="/unmeasured-storage";if(table.recommend(unknown)!=limbforge::BackendRecommendation::unknown)return 5;
++checked;}
{limbforge::DispatchKey key{"complex_gemm/complex_composed",352,1,66,3800,130,18,false,"WolfNum 1.3.1 / Apple M5 Max / Release / busy-host library; no idle consumer acceptance; measured=Apple M5 Max / 1.3.1 / complex-gemm-contracts; binary=a8a40fbbf9a3290cd200d4c24d0d632554c6d739b168c55cf2892a9e15725d11; archive=c9dd6ebed304b6f74258f8ea0aebb4c29e1b205fff4deb115254ee5652ff9d24; clock=cpu-interleaved; cpu=MPFR/complex_composed; active_workers=18; load_series=37aa78f5da509750a37950a1c3038e1c385d4d66cefa2e7ef0f244f0169b9411; all_loadavg_below_4=false; scope=numeric-array-library; bridge=excluded",true};
if(table.recommend(key)!=limbforge::BackendRecommendation::gpu)return 1;
auto unknown=key;unknown.bits+=32;if(table.recommend(unknown)!=limbforge::BackendRecommendation::unknown)return 2;
unknown=key;unknown.profile+="/unmeasured";if(table.recommend(unknown)!=limbforge::BackendRecommendation::unknown)return 3;
unknown=key;unknown.k+=1;if(table.recommend(unknown)!=limbforge::BackendRecommendation::unknown)return 4;
unknown=key;unknown.resident=!key.resident;unknown.profile+="/unmeasured-storage";if(table.recommend(unknown)!=limbforge::BackendRecommendation::unknown)return 5;
++checked;}
{limbforge::DispatchKey key{"complex_gemm/complex_fused",352,1,66,3800,130,18,true,"WolfNum 1.3.1 / Apple M5 Max / Release / busy-host library; no idle consumer acceptance; measured=Apple M5 Max / 1.3.1 / complex-gemm-contracts; binary=a8a40fbbf9a3290cd200d4c24d0d632554c6d739b168c55cf2892a9e15725d11; archive=c9dd6ebed304b6f74258f8ea0aebb4c29e1b205fff4deb115254ee5652ff9d24; clock=cpu-interleaved; cpu=MPFR/complex_fused; active_workers=18; load_series=37aa78f5da509750a37950a1c3038e1c385d4d66cefa2e7ef0f244f0169b9411; all_loadavg_below_4=false; scope=numeric-array-library; bridge=excluded",true};
if(table.recommend(key)!=limbforge::BackendRecommendation::gpu)return 1;
auto unknown=key;unknown.bits+=32;if(table.recommend(unknown)!=limbforge::BackendRecommendation::unknown)return 2;
unknown=key;unknown.profile+="/unmeasured";if(table.recommend(unknown)!=limbforge::BackendRecommendation::unknown)return 3;
unknown=key;unknown.k+=1;if(table.recommend(unknown)!=limbforge::BackendRecommendation::unknown)return 4;
unknown=key;unknown.resident=!key.resident;unknown.profile+="/unmeasured-storage";if(table.recommend(unknown)!=limbforge::BackendRecommendation::unknown)return 5;
++checked;}
{limbforge::DispatchKey key{"complex_gemm/exact_real_embedding",352,1,66,3800,130,18,false,"WolfNum 1.3.1 / Apple M5 Max / Release / busy-host library; no idle consumer acceptance; measured=Apple M5 Max / 1.3.1 / complex-gemm-contracts; binary=a8a40fbbf9a3290cd200d4c24d0d632554c6d739b168c55cf2892a9e15725d11; archive=c9dd6ebed304b6f74258f8ea0aebb4c29e1b205fff4deb115254ee5652ff9d24; clock=cpu-interleaved; cpu=MPFR/exact_real_embedding; active_workers=18; load_series=37aa78f5da509750a37950a1c3038e1c385d4d66cefa2e7ef0f244f0169b9411; all_loadavg_below_4=false; scope=numeric-array-library; bridge=excluded",true};
if(table.recommend(key)!=limbforge::BackendRecommendation::gpu)return 1;
auto unknown=key;unknown.bits+=32;if(table.recommend(unknown)!=limbforge::BackendRecommendation::unknown)return 2;
unknown=key;unknown.profile+="/unmeasured";if(table.recommend(unknown)!=limbforge::BackendRecommendation::unknown)return 3;
unknown=key;unknown.k+=1;if(table.recommend(unknown)!=limbforge::BackendRecommendation::unknown)return 4;
unknown=key;unknown.resident=!key.resident;unknown.profile+="/unmeasured-storage";if(table.recommend(unknown)!=limbforge::BackendRecommendation::unknown)return 5;
++checked;}
{limbforge::DispatchKey key{"complex_gemm/gauss_three_real_composed",352,1,66,3800,130,18,false,"WolfNum 1.3.1 / Apple M5 Max / Release / busy-host library; no idle consumer acceptance; measured=Apple M5 Max / 1.3.1 / complex-gemm-contracts; binary=a8a40fbbf9a3290cd200d4c24d0d632554c6d739b168c55cf2892a9e15725d11; archive=c9dd6ebed304b6f74258f8ea0aebb4c29e1b205fff4deb115254ee5652ff9d24; clock=cpu-interleaved; cpu=MPFR/gauss_three_real_composed; active_workers=18; load_series=37aa78f5da509750a37950a1c3038e1c385d4d66cefa2e7ef0f244f0169b9411; all_loadavg_below_4=false; scope=numeric-array-library; bridge=excluded",true};
if(table.recommend(key)!=limbforge::BackendRecommendation::gpu)return 1;
auto unknown=key;unknown.bits+=32;if(table.recommend(unknown)!=limbforge::BackendRecommendation::unknown)return 2;
unknown=key;unknown.profile+="/unmeasured";if(table.recommend(unknown)!=limbforge::BackendRecommendation::unknown)return 3;
unknown=key;unknown.k+=1;if(table.recommend(unknown)!=limbforge::BackendRecommendation::unknown)return 4;
unknown=key;unknown.resident=!key.resident;unknown.profile+="/unmeasured-storage";if(table.recommend(unknown)!=limbforge::BackendRecommendation::unknown)return 5;
++checked;}
{limbforge::DispatchKey key{"complex_gemm/complex_composed",352,100000,4,4,4,18,false,"WolfNum 1.3.1 / Apple M5 Max / Release / busy-host library; no idle consumer acceptance; measured=Apple M5 Max / 1.3.1 / complex-gemm-contracts; binary=a8a40fbbf9a3290cd200d4c24d0d632554c6d739b168c55cf2892a9e15725d11; archive=c9dd6ebed304b6f74258f8ea0aebb4c29e1b205fff4deb115254ee5652ff9d24; clock=cpu-interleaved; cpu=MPFR/complex_composed; active_workers=18; load_series=a39b4661361e14ed00271394d83a8dcf5253b877dbd99c903d2c1620a5c22ed7; all_loadavg_below_4=false; scope=numeric-array-library; bridge=excluded",true};
if(table.recommend(key)!=limbforge::BackendRecommendation::gpu)return 1;
auto unknown=key;unknown.bits+=32;if(table.recommend(unknown)!=limbforge::BackendRecommendation::unknown)return 2;
unknown=key;unknown.profile+="/unmeasured";if(table.recommend(unknown)!=limbforge::BackendRecommendation::unknown)return 3;
unknown=key;unknown.k+=1;if(table.recommend(unknown)!=limbforge::BackendRecommendation::unknown)return 4;
unknown=key;unknown.resident=!key.resident;unknown.profile+="/unmeasured-storage";if(table.recommend(unknown)!=limbforge::BackendRecommendation::unknown)return 5;
++checked;}
{limbforge::DispatchKey key{"complex_gemm/complex_fused",352,100000,4,4,4,18,true,"WolfNum 1.3.1 / Apple M5 Max / Release / busy-host library; no idle consumer acceptance; measured=Apple M5 Max / 1.3.1 / complex-gemm-contracts; binary=a8a40fbbf9a3290cd200d4c24d0d632554c6d739b168c55cf2892a9e15725d11; archive=c9dd6ebed304b6f74258f8ea0aebb4c29e1b205fff4deb115254ee5652ff9d24; clock=cpu-interleaved; cpu=MPFR/complex_fused; active_workers=18; load_series=b29eb5db0917a139f918d4575cab2fa6b482ba463762c5f9ac2dc142d367c5f0; all_loadavg_below_4=false; scope=numeric-array-library; bridge=excluded",true};
if(table.recommend(key)!=limbforge::BackendRecommendation::gpu)return 1;
auto unknown=key;unknown.bits+=32;if(table.recommend(unknown)!=limbforge::BackendRecommendation::unknown)return 2;
unknown=key;unknown.profile+="/unmeasured";if(table.recommend(unknown)!=limbforge::BackendRecommendation::unknown)return 3;
unknown=key;unknown.k+=1;if(table.recommend(unknown)!=limbforge::BackendRecommendation::unknown)return 4;
unknown=key;unknown.resident=!key.resident;unknown.profile+="/unmeasured-storage";if(table.recommend(unknown)!=limbforge::BackendRecommendation::unknown)return 5;
++checked;}
{limbforge::DispatchKey key{"complex_gemm/gauss_three_real_composed",352,100000,4,4,4,18,false,"WolfNum 1.3.1 / Apple M5 Max / Release / busy-host library; no idle consumer acceptance; measured=Apple M5 Max / 1.3.1 / complex-gemm-contracts; binary=a8a40fbbf9a3290cd200d4c24d0d632554c6d739b168c55cf2892a9e15725d11; archive=c9dd6ebed304b6f74258f8ea0aebb4c29e1b205fff4deb115254ee5652ff9d24; clock=cpu-interleaved; cpu=MPFR/gauss_three_real_composed; active_workers=18; load_series=a1fd4209fa35c28e7959340ffe3b9de44043bb12280711c714845326d74ca5d9; all_loadavg_below_4=false; scope=numeric-array-library; bridge=excluded",true};
if(table.recommend(key)!=limbforge::BackendRecommendation::gpu)return 1;
auto unknown=key;unknown.bits+=32;if(table.recommend(unknown)!=limbforge::BackendRecommendation::unknown)return 2;
unknown=key;unknown.profile+="/unmeasured";if(table.recommend(unknown)!=limbforge::BackendRecommendation::unknown)return 3;
unknown=key;unknown.k+=1;if(table.recommend(unknown)!=limbforge::BackendRecommendation::unknown)return 4;
unknown=key;unknown.resident=!key.resident;unknown.profile+="/unmeasured-storage";if(table.recommend(unknown)!=limbforge::BackendRecommendation::unknown)return 5;
++checked;}
{limbforge::DispatchKey key{"complex_gemm/complex_composed",352,10000,4,4,4,18,false,"WolfNum 1.3.1 / Apple M5 Max / Release / busy-host library; no idle consumer acceptance; measured=Apple M5 Max / 1.3.1 / complex-gemm-contracts; binary=a8a40fbbf9a3290cd200d4c24d0d632554c6d739b168c55cf2892a9e15725d11; archive=c9dd6ebed304b6f74258f8ea0aebb4c29e1b205fff4deb115254ee5652ff9d24; clock=cpu-interleaved; cpu=MPFR/complex_composed; active_workers=18; load_series=cf36b519773369761436edcc1e6cf0b8ed6cd67a3169bee720718f547db973bd; all_loadavg_below_4=false; scope=numeric-array-library; bridge=excluded",true};
if(table.recommend(key)!=limbforge::BackendRecommendation::gpu)return 1;
auto unknown=key;unknown.bits+=32;if(table.recommend(unknown)!=limbforge::BackendRecommendation::unknown)return 2;
unknown=key;unknown.profile+="/unmeasured";if(table.recommend(unknown)!=limbforge::BackendRecommendation::unknown)return 3;
unknown=key;unknown.k+=1;if(table.recommend(unknown)!=limbforge::BackendRecommendation::unknown)return 4;
unknown=key;unknown.resident=!key.resident;unknown.profile+="/unmeasured-storage";if(table.recommend(unknown)!=limbforge::BackendRecommendation::unknown)return 5;
++checked;}
{limbforge::DispatchKey key{"complex_gemm/complex_fused",352,10000,4,4,4,18,true,"WolfNum 1.3.1 / Apple M5 Max / Release / busy-host library; no idle consumer acceptance; measured=Apple M5 Max / 1.3.1 / complex-gemm-contracts; binary=a8a40fbbf9a3290cd200d4c24d0d632554c6d739b168c55cf2892a9e15725d11; archive=c9dd6ebed304b6f74258f8ea0aebb4c29e1b205fff4deb115254ee5652ff9d24; clock=cpu-interleaved; cpu=MPFR/complex_fused; active_workers=18; load_series=b66976b73487dd5fab034e80968d825c2904eeeaca58e1a3d4631f2018ecbb2b; all_loadavg_below_4=false; scope=numeric-array-library; bridge=excluded",true};
if(table.recommend(key)!=limbforge::BackendRecommendation::gpu)return 1;
auto unknown=key;unknown.bits+=32;if(table.recommend(unknown)!=limbforge::BackendRecommendation::unknown)return 2;
unknown=key;unknown.profile+="/unmeasured";if(table.recommend(unknown)!=limbforge::BackendRecommendation::unknown)return 3;
unknown=key;unknown.k+=1;if(table.recommend(unknown)!=limbforge::BackendRecommendation::unknown)return 4;
unknown=key;unknown.resident=!key.resident;unknown.profile+="/unmeasured-storage";if(table.recommend(unknown)!=limbforge::BackendRecommendation::unknown)return 5;
++checked;}
{limbforge::DispatchKey key{"complex_gemm/gauss_three_real_composed",352,10000,4,4,4,18,false,"WolfNum 1.3.1 / Apple M5 Max / Release / busy-host library; no idle consumer acceptance; measured=Apple M5 Max / 1.3.1 / complex-gemm-contracts; binary=a8a40fbbf9a3290cd200d4c24d0d632554c6d739b168c55cf2892a9e15725d11; archive=c9dd6ebed304b6f74258f8ea0aebb4c29e1b205fff4deb115254ee5652ff9d24; clock=cpu-interleaved; cpu=MPFR/gauss_three_real_composed; active_workers=18; load_series=1e8daedacd47633c66cf0bd224a4cb11eb7116fb42cb364016c59ae29eca4853; all_loadavg_below_4=false; scope=numeric-array-library; bridge=excluded",true};
if(table.recommend(key)!=limbforge::BackendRecommendation::gpu)return 1;
auto unknown=key;unknown.bits+=32;if(table.recommend(unknown)!=limbforge::BackendRecommendation::unknown)return 2;
unknown=key;unknown.profile+="/unmeasured";if(table.recommend(unknown)!=limbforge::BackendRecommendation::unknown)return 3;
unknown=key;unknown.k+=1;if(table.recommend(unknown)!=limbforge::BackendRecommendation::unknown)return 4;
unknown=key;unknown.resident=!key.resident;unknown.profile+="/unmeasured-storage";if(table.recommend(unknown)!=limbforge::BackendRecommendation::unknown)return 5;
++checked;}
std::cout<<checked<<" measured GEMM keys and unknown width/profile/shape/storage checks passed\n";return 0;}
