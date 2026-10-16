# The Mill workflow

## Evidence and open checks

Read-only reference: sibling J1J2 repository, inspected October 14, 2026:
`configs/millcli.yaml`, `cluster/mill_files/batch/template.sub`,
`configs/validation/honeycomb/mill_stripe_smoke.yaml`, and
`docs/DATA_WORKFLOW.md`. Its flow is source sync -> submit -> status ->
pull raw -> derive -> plot, using SSH, Slurm, and scratch raw storage.

[University quick start](https://itrss.mst.edu/cluster/quickstart/)
documents SSH, file transfer, sbatch, and squeue. The existing J1J2 host
is `mill.mst.edu`. No remote login or phi4 job has been attempted here.

J1J2 uses `openmpi/4.1.6/gcc/12.2.0`; its recent smoke config uses partition
`requeue`. Those are local reference observations, not confirmation of
current module availability, partition access, limits, or phi4 requirements.
A serial phi4 process does not require MPI. Verify g++, CMake, modules,
partition/account policy, scratch retention, quotas, and queue limits before
the first campaign. Never infer the user's CPU-hour budget from J1J2.

## Separate project paths

Proposed locations (must be confirmed on Mill):

```text
source snapshots: $HOME/phi4/releases/<source-id>/
Linux builds:     $HOME/phi4/builds/<source-id>/
job logs:         $HOME/phi4/logs/
raw runs:         /share/ceph/scratch/$USER/phi4/raw/<campaign>/<run_id>/
durable results:  $HOME/phi4/results/<campaign>/
local raw mirror: data/raw/<campaign>/<run_id>/
```

Environment variables above are expanded by the remote shell, not treated
as literal directory names or expanded to the Windows user's home.
Keep phi4 separate from J1J2. Scratch is working storage; confirm retention
and back up required raw measurements before considering it an archive.

## First deployment sequence (not executed)

1. Check SSH/access and modules with read-only commands. Keep credentials
   and host keys in normal SSH configuration, never repo files.
2. Sync a clean source snapshot plus hashes, excluding local builds, data,
   results, `.venv`, `.git`, and secrets. Do not sync a mutable source tree
   underneath running jobs. Native Windows can use OpenSSH scp/SFTP;
   use rsync from a proper shell environment if it is available later.
3. Build and run C++ validation in a short permitted allocation. Keep
   simulation workloads off the login node; use the site's build policy.
4. Submit one tiny smoke run and pull it. Verify exit status, metadata,
   completeness, parser, and Windows/Mill numerical consistency.
5. Benchmark sweeps and measurements. Estimate per-chain runtime and
   total CPU-hours. Approve a concrete campaign budget before production.
6. Render/dry-run a manifest, inspect it, then submit bounded arrays.
   One task, one CPU, one independent chain; array concurrency must obey
   site policy and the authorized budget. Choose memory/time from pilots.
7. Record job IDs and manifest hash. Inspect queue and accounting; pull
   completed and failed-run diagnostics separately. Check walltime exits.
8. Derive and plot locally, preserving raw samples. Update the ledger.

Useful read-only remote commands after connecting: `module avail`,
`g++ --version`, `cmake --version`, `sinfo`, `squeue -u "$USER"`,
and `sacct -j JOB_ID --format=JobID,State,ExitCode,Elapsed,MaxRSS`.
These have not been run against Mill in this task.

## Slurm contract for later implementation

Render an explicit partition/account if required, walltime, memory,
`--ntasks=1`, `--cpus-per-task=1`, a capped array, and stdout/stderr paths.
Create the logs directory *before* sbatch; Slurm opens log files before
the body runs. Map each array index to exactly one manifest row; validate
index bounds, config hash, binary identity, and unique output directory.

Use fail-fast shell behavior and explicit error traps. Write SUCCESS only
after outputs are closed and validated. Requeue must not append duplicated
measurements or overwrite a completed run. Until checkpoint resume is
implemented, each attempt gets a new identity and is a fresh chain.
Preserve failure logs; do not copy J1J2's automatic cleanup behavior.

A network/sandbox error is not a reason to change credentials or retry
continuously. Report the limitation, obtain tool escalation when available,
and continue local work. No inherited J1J2 authorization grants permission
to submit phi4 campaigns, cancel jobs, or remove remote files.
