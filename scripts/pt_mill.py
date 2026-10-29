"""Bounded Mill deployment/submission for a single MPI temperature ladder."""

import argparse
import json
from pathlib import Path

import mill
import run as single
import run_pt

REPO = Path(__file__).resolve().parents[1]


def build(args):
    root = f"{mill.REMOTE_BASE}/{mill.checked_name(args.release)}"
    attempt = mill.checked_name(args.attempt)
    script = f"""set -euo pipefail
exec 9>/home/{mill.USER}/phi4/submit.lock
flock -x 9
count=$(squeue -h -r -u {mill.USER} -o '%i' | wc -l)
(( count < 29 )) || exit 2
test ! -f {root}/build/mill-pt/VALIDATED
mkdir {root}/cluster/pt-build-attempt-{attempt}
cat > {root}/cluster/pt-build-attempt-{attempt}/build_pt.sh <<'BUILD'
#!/bin/bash
set -euo pipefail
set +eu
source /etc/profile >/dev/null 2>&1
set -eu
module load openmpi/4.1.6/gcc/12.2.0
cd {root}
mkdir -p build/mill-pt
mpic++ -std=c++17 -O3 -Wall -Wextra -Wpedantic -Iinclude src/pt.cpp -o build/mill-pt/phi4_pt_serial
mpic++ -std=c++17 -O3 -Wall -Wextra -Wpedantic -DPHI4_MPI=1 -Iinclude src/pt.cpp -o build/mill-pt/phi4_pt
mpic++ -std=c++17 -O3 -Iinclude -DPHI4_COMPILER='"GCC"' -DPHI4_BUILD_TYPE='"Release"' src/main.cpp -o build/mill-pt/phi4
mpic++ -std=c++17 -O2 -Iinclude tests/cpp/test_tempering.cpp -o build/mill-pt/test_tempering
mpic++ -std=c++17 -O2 -Iinclude tests/cpp/test_model.cpp -o build/mill-pt/test_model
mpic++ -std=c++17 -O2 -Iinclude tests/cpp/test_distribution.cpp -o build/mill-pt/test_distribution
build/mill-pt/test_model
build/mill-pt/test_distribution
build/mill-pt/test_tempering
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
python3.11 tests/python/test_pt.py build/mill-pt/phi4_pt_serial build/mill-pt/phi4 build/mill-pt/phi4_pt
sha256sum build/mill-pt/phi4_pt build/mill-pt/phi4_pt_serial > build/mill-pt/binary.sha256
mpic++ --showme:version > build/mill-pt/compiler.txt
touch build/mill-pt/VALIDATED
BUILD
sbatch --parsable --partition=requeue --account=general --ntasks=3 --cpus-per-task=1 --mem=4G --time=10 --no-requeue --job-name=phi4-pt-validation --output={root}/cluster/logs/pt-build-%j.out --error={root}/cluster/logs/pt-build-%j.err {root}/cluster/pt-build-attempt-{attempt}/build_pt.sh
"""
    result = mill.remote("bash -s", input=script, text=True, capture_output=True)
    print(result.stdout, end="")


def plan(args):
    mill.checked_name(args.release)
    mill.checked_name(args.campaign)
    if not 1 <= args.minutes <= 1440:
        raise ValueError("Invalid walltime")
    folder = REPO / "cluster/manifests" / args.campaign
    text = mill.config_text(
        args.L,
        args.temperatures[0],
        args.start,
        args.seed,
        args.campaign,
        0,
        args.adaptation,
        args.settling,
        args.measurements,
        args.interval,
    )
    text += (
        "\n[tempering]\ntemperatures = "
        + json.dumps(args.temperatures)
        + "\nswap_every_sweeps = "
        + str(args.swap_every)
        + "\n"
    )
    import tomllib

    run_pt.validate(tomllib.loads(text))
    folder.mkdir(parents=True, exist_ok=False)
    (folder / "ladder.toml").write_text(text, encoding="utf-8", newline="\n")
    record = {
        "schema_version": 1,
        "release": args.release,
        "campaign": args.campaign,
        "ranks": len(args.temperatures),
        "minutes": args.minutes,
        "maximum_cpu_hours": len(args.temperatures) * args.minutes / 60,
        "config_sha256": single.sha256(folder / "ladder.toml"),
        "created_utc": single.utc(),
        "method": "MPI temperature-label replica exchange; Metropolis local updates; no exchanges when swap_every=0",
    }
    (folder / "pt_plan.json").write_text(json.dumps(record, indent=2) + "\n")
    print(json.dumps(record, indent=2))


def submit(args):
    path = Path(args.plan).resolve()
    data = json.loads(path.read_text())
    release = mill.checked_name(data["release"])
    campaign = mill.checked_name(data["campaign"])
    if not 1 <= data["ranks"] <= 256 or not 1 <= data["minutes"] <= 1440:
        raise ValueError("Invalid plan resources")
    maximum = data["ranks"] * data["minutes"] / 60
    if maximum > args.budget_cpu_hours or not 0 < args.budget_cpu_hours < float("inf"):
        raise ValueError("Explicit CPU-hour cap exceeded")
    if single.sha256(path.parent / "ladder.toml") != data["config_sha256"]:
        raise ValueError("Config hash mismatch")
    if (path.parent / "submission.json").exists():
        raise ValueError("Already submitted; retries need a new campaign")
    root = f"{mill.REMOTE_BASE}/{release}"
    remote_folder = f"{root}/cluster/manifests/{campaign}"
    mill.remote(f"set -eu; mkdir -p {root}/cluster/manifests; mkdir {remote_folder}")
    import subprocess

    subprocess.run(
        [
            "scp",
            "-o",
            "BatchMode=yes",
            str(path),
            str(path.parent / "ladder.toml"),
            f"{mill.DESTINATION}:{remote_folder}/",
        ],
        check=True,
    )
    script = f"""set -euo pipefail
exec 9>/home/{mill.USER}/phi4/submit.lock
flock -x 9
count=$(squeue -h -r -u {mill.USER} -o '%i' | wc -l)
(( count < 29 )) || {{ echo 'Account job cap exceeded' >&2; exit 2; }}
test -f {root}/build/mill-pt/VALIDATED
cd {root}
sha256sum -c build/mill-pt/binary.sha256 >/dev/null
cat > {remote_folder}/job.sh <<'JOB'
#!/bin/bash
set -euo pipefail
set +eu
source /etc/profile >/dev/null 2>&1
set -eu
module load openmpi/4.1.6/gcc/12.2.0
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
cd {root}
python3.11 scripts/pt_mill.py worker cluster/manifests/{campaign}/pt_plan.json
JOB
sbatch --parsable --partition=requeue --account=general --ntasks={data["ranks"]} --cpus-per-task=1 --mem-per-cpu=512M --time={data["minutes"]} --no-requeue --job-name=phi4-{campaign} --output={root}/cluster/logs/%x-%j.out --error={root}/cluster/logs/%x-%j.err {remote_folder}/job.sh
"""
    record = {
        "status": "SUBMITTING",
        "submitted_utc": single.utc(),
        "plan_sha256": single.sha256(path),
        "maximum_cpu_hours": maximum,
        "ranks": data["ranks"],
    }
    record_path = path.parent / "submission.json"
    record_path.write_text(json.dumps(record, indent=2))
    try:
        result = mill.remote("bash -s", input=script, text=True, capture_output=True)
        job = result.stdout.strip().splitlines()[-1]
        if not job.split(";")[0].isdigit():
            raise ValueError("Unrecognized sbatch output")
        record.update(status="SUBMITTED", job_id=job, remote_root=root)
    except Exception as error:
        record.update(status="SUBMISSION_UNCERTAIN", error=str(error))
        raise
    finally:
        record_path.write_text(json.dumps(record, indent=2) + "\n")
    print(json.dumps(record, indent=2))


def worker(args):
    path = Path(args.plan).resolve()
    data = json.loads(path.read_text())
    source = json.loads((REPO / "release.json").read_text())
    if source["release"] != data["release"]:
        raise ValueError("Wrong source snapshot")
    for name, digest in source["files"].items():
        if single.sha256(REPO / name) != digest:
            raise ValueError("Source changed after deployment: " + name)
    for line in (REPO / "build/mill-pt/binary.sha256").read_text().splitlines():
        digest, name = line.split(maxsplit=1)
        if single.sha256(REPO / name) != digest:
            raise ValueError("Binary changed after validation")
    config = path.parent / "ladder.toml"
    if single.sha256(config) != data["config_sha256"]:
        raise ValueError("Config changed")
    run_pt.launch(config, REPO / "build/mill-pt/phi4_pt", "mpi")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    p = sub.add_parser("build")
    p.add_argument("release")
    p.add_argument("--attempt", default="initial")
    p = sub.add_parser("plan")
    p.add_argument("release")
    p.add_argument("campaign")
    p.add_argument("--L", type=int, required=True)
    p.add_argument("--temperatures", type=float, nargs="+", required=True)
    p.add_argument(
        "--start",
        choices=("random", "ordered_positive", "ordered_negative"),
        default="random",
    )
    p.add_argument("--seed", type=int, default=544000000)
    p.add_argument("--adaptation", type=int, default=2000)
    p.add_argument("--settling", type=int, default=10000)
    p.add_argument("--measurements", type=int, default=100000)
    p.add_argument("--interval", type=int, default=5)
    p.add_argument("--swap-every", type=int, default=5)
    p.add_argument("--minutes", type=int, default=10)
    p = sub.add_parser("submit")
    p.add_argument("plan")
    p.add_argument("--budget-cpu-hours", type=float, required=True)
    p = sub.add_parser("worker")
    p.add_argument("plan")
    args = parser.parse_args()
    if args.command in ("build", "submit") and not mill.SITE_PATH.exists():
        parser.error("Configure configs/mill.local.toml first")
    globals()[args.command](args)


if __name__ == "__main__":
    main()
