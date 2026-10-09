# Section 9 P1.6: intermediate reuse candidate

Base: `e91a215` / 1.1.0. Branch: `round-section9-batched-workspaces`.
Accepted in 1.2.0 after combined focused normal/shader and package checks.

No shader arithmetic changed. `product3` caches its intermediate; exact normals cache
augmentation and Gram separately for each live call. Completion closures own slots,
with weak detached bookkeeping after release. Allocations stay on the unit's engine.
Idle selection also checks buffer busy flags to avoid the interval between clearing
completion closures and releasing submission resources.

Focused checks: `section9_p16_workspace_final_smoke.txt` and
`section9_p16_workspace_final_validation.txt`, both pass. Coverage includes original MPFR
arithmetic checks, warmed capacity reuse, unsubmitted/submitted ownership, busy release,
abandoned batches, empty-product compatibility, foreign-engine rejection, distinct forced-fallback repairs of two
normal equations in one batch, and algebra-object destruction before submission.
This is not a new full arithmetic audit or a performance measurement.

Warm reuse removes repeated intermediate Metal buffer creation. Host encoding still
allocates completion/bookkeeping objects. Pending concurrency needs one slot per call;
retention increases idle memory until explicitly released. No timing claim is accepted.

Version/package checks pass 2/2 (`section9_p16_version_package.txt`).

Combined release records: `section9_host_api_1_2_*`, including source/archive hashes
and the explicit focused validation scope. No full arithmetic sweep was repeated
for these unchanged shaders.
