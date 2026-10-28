# The Mill workflow

The Mill commands are in `scripts/mill.py`. The sampler remains sequential
single-site Metropolis. Parallel work consists of independent chains at
separate lattice sizes, temperatures and seeds, scheduled as Slurm array
workers. Each worker uses one CPU and runs its assigned rows in sequence.

## Site reference and settings

The site reference is the sibling J1J2 repository, read without modification:
`configs/millcli.yaml`, `cluster/mill_files/batch/template.sub`,
`configs/validation/honeycomb/mill_stripe_smoke.yaml`, and
`docs/DATA_WORKFLOW.md`. Live checks on October 28 confirmed SSH access,
partition `requeue`, account `general`, GCC 8.5 and `/usr/bin/python3.11`.
The default Python is 3.6 and cannot run the launcher. No MPI is needed.

See the [Mill documentation](https://docs.itrss.umsystem.edu/pub/hpc/mill)
for current policies. Compilation, tests and sampling run in scheduled compute
allocations. Login-node operations are transfers, queue queries and submission.

Copy `configs/mill.example.toml` to ignored `configs/mill.local.toml` and set
the SSO ID. SSH keys remain in the normal SSH configuration.

## Commands

From the repository root:

```text
python scripts/mill.py deploy
python scripts/mill.py build RELEASE
python scripts/mill.py status
```

Deployment creates a new source snapshot, refuses to overwrite an existing
release and verifies the transferred archive checksum. Read the build job's
logs before proceeding. The compute job builds with C++17/O3 (without fast
math), then runs the independent model, distribution and workflow checks.
A VALIDATED marker is written only if all commands succeed.

Prepare a short pilot:

```text
python scripts/mill.py plan RELEASE example-pilot --sizes 8 12 --temperatures 15.25 15.35 15.45 --tasks 12 --minutes 10
python scripts/mill.py submit cluster/manifests/example-pilot/manifest.json --budget-cpu-hours 2
python scripts/mill.py pull RELEASE example-pilot
```

`plan` only writes local configurations and a manifest. The printed maximum
CPU-hours is worker count times walltime, including all sequential rows.
Submission requires an explicit budget at least that large. User approval
of a campaign remains separate from the command-line budget argument.

The account limit is 29 queued/running jobs, counting all the user's work,
including J1J2 and every expanded array element. Submission checks `squeue -r`
and uses a per-user lock to serialize this tool's submissions. Each batch has
at most 27 workers. Other submission clients must also obey the account cap;
the tool cannot reserve future slots against unrelated concurrent clients.
The pilot maximum observed total was 27 jobs, including two J1J2 jobs.

## Storage and integrity

Scratch releases live under `/share/ceph/scratch/SSO/phi4/releases/RELEASE`.
Each release contains sources, its build, job logs and `data/raw/CAMPAIGN`.
The complete release is on scratch so the launcher's existing repository-bound
output validation remains intact. A source archive and release hash manifest
are also retained under `/home/SSO/phi4/releases/RELEASE`.

Each worker verifies source-file hashes, binary hash and each row's config hash
before sampling. The launcher retains raw histories, configuration, source
archive, seed, binary checksum, timings, success/failure metadata and output
checksums. Independent starts never continue a preceding temperature's field.
Every attempt needs a unique campaign/run identity; existing raw outputs are
never overwritten. Requeue is disabled because restart checkpoints do not exist.
Logs and failed attempts are retained. There is no automatic cleanup or retry.

`pull` refuses an existing local campaign. For a separate recovery copy use
`--destination data/raw/NEW_NAME`. Pull after all workers finish; incomplete
copies remain explicit and are not silently repaired. Scratch and home are
not backed up by Mill. Download required raw measurements to local storage.

## Validation limits

The October 28 release is `34f3b89caaf49088c7bc`. It includes the working-tree
single-temperature `main.cpp` present at deployment. The existing uncommitted
removal of temperature-scan support was preserved. The legacy scan test fails;
model, distribution and single-chain workflow validation pass on Mill.
The new local scientific analysis was added after that sampler snapshot;
analysis outputs record their own source hashes.

After reviewing the October 28 pilot benchmarks, the user approved the
production-proposal-20261028-b campaign with a 50 CPU-hour ceiling. Array
2633589 was submitted with 25 one-CPU workers and two-hour limits. No background monitor or
additional campaign is started by `status`, `pull` or analysis.

`plan --replicas 2` creates two independent seeds for each requested start,
so the default random/ordered pair becomes four chains per ensemble. Production
proposal generation is not permission to submit. The current unsubmitted
proposal is production-proposal-20261028-b; earlier proposal grids are superseded.
