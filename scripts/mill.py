"""Bounded Mill submission for independent, single-temperature Metropolis chains.

No background monitor, cancellation, cleanup, or automatic retry is performed.
"""

import argparse
import hashlib
import io
import json
import re
import shlex
import subprocess
import tarfile
import tomllib
from datetime import datetime, timezone
from pathlib import Path

import run as launcher

REPO = Path(__file__).resolve().parents[1]
SITE_PATH = REPO / "configs/mill.local.toml"
SITE = tomllib.loads(SITE_PATH.read_text()) if SITE_PATH.exists() else {}
HOST = SITE.get("remote", {}).get("host", "mill.mst.edu")
USER = SITE.get("remote", {}).get("user", "REPLACE_WITH_SSO_ID")
REMOTE_BASE = f"/share/ceph/scratch/{USER}/phi4/releases"
HOME_BASE = f"/home/{USER}/phi4/releases"
DESTINATION = f"{USER}@{HOST}"


def checked_name(value):
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]{0,95}", value):
        raise ValueError(
            "Use a short name containing letters, numbers, underscores or hyphens"
        )
    return value


def remote(command, **kwargs):
    text_mode = kwargs.pop("text", False)
    if isinstance(kwargs.get("input"), str):
        kwargs["input"] = kwargs["input"].replace("\r\n", "\n").encode("utf-8")
    result = subprocess.run(
        ["ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=20", DESTINATION, command],
        check=True,
        **kwargs,
    )
    if text_mode:
        if result.stdout is not None:
            result.stdout = result.stdout.decode("utf-8")
        if result.stderr is not None:
            result.stderr = result.stderr.decode("utf-8")
    return result


def queue_count(output):
    """squeue -r expands array elements, including pending elements."""
    return len([line for line in output.splitlines() if line.strip()])


def slots(output, requested, cap=29):
    if type(requested) is not int or requested < 1 or not 1 <= cap <= 29:
        raise ValueError("Require positive task count and an account cap of at most 29")
    count = queue_count(output)
    if count + requested > cap:
        raise ValueError(f"{count} existing jobs + {requested} requested exceeds {cap}")
    return count


def package():
    """Capture the working tree without altering any existing edits."""
    paths = set()
    for pattern in (
        "include/**/*.hpp",
        "src/*.cpp",
        "scripts/*.py",
        "analysis/**/*.py",
        "tests/**/*.py",
        "tests/**/*.cpp",
        "configs/**/*.toml",
        "docs/*.md",
        "CMakeLists.txt",
        "CMakePresets.json",
        "README.md",
    ):
        for path in REPO.glob(pattern):
            if ".local." not in path.name and "__pycache__" not in path.parts:
                paths.add(path)
    files = {
        p.relative_to(REPO).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest()
        for p in sorted(paths)
    }
    identity = hashlib.sha256(json.dumps(files, sort_keys=True).encode()).hexdigest()[
        :20
    ]
    folder = REPO / "build" / "mill" / identity
    folder.mkdir(parents=True, exist_ok=True)
    archive_path = folder / "source.tar.gz"
    with tarfile.open(archive_path, "w:gz") as archive:
        for path in sorted(paths):
            archive.add(
                path, arcname=path.relative_to(REPO).as_posix(), recursive=False
            )
        payload = json.dumps({"files": files, "release": identity}, indent=2).encode()
        info = tarfile.TarInfo("release.json")
        info.size = len(payload)
        archive.addfile(info, io.BytesIO(payload))
    return identity, archive_path


def deploy():
    identity, archive = package()
    destination = f"{REMOTE_BASE}/{identity}"
    durable = f"{HOME_BASE}/{identity}"
    digest = launcher.sha256(archive)
    # Atomic mkdir refuses replacement of an existing source snapshot.
    remote(
        f"set -eu; mkdir -p {shlex.quote(REMOTE_BASE)} {shlex.quote(HOME_BASE)}; mkdir {shlex.quote(destination)}; mkdir {shlex.quote(durable)}"
    )
    subprocess.run(
        [
            "scp",
            "-o",
            "BatchMode=yes",
            str(archive),
            f"{DESTINATION}:{durable}/source.tar.gz",
        ],
        check=True,
    )
    remote(
        f"set -eu; echo '{digest}  {durable}/source.tar.gz' | sha256sum -c -; tar -xzf {durable}/source.tar.gz -C {destination}; mkdir -p {destination}/cluster/logs; cp {destination}/release.json {durable}/release.json"
    )
    print(
        json.dumps(
            {
                "release": identity,
                "source_archive_sha256": digest,
                "remote_root": destination,
            },
            indent=2,
        )
    )


def config_text(
    L,
    temperature,
    start,
    seed,
    campaign,
    row,
    adaptation,
    settling,
    measurements,
    interval,
):
    return f'''schema_version = 1
[model]
name = "clean_phi4_3d"
L = {L}
temperature = {float(temperature)}
J = 1.0
r = -1.0
u = 1.0
boundary = "periodic"
[sampling]
start = "{start}"
seed = {seed}
update_order = "sequential"
step_size = 1.0
adaptation_sweeps = {adaptation}
equilibration_sweeps = {settling}
measurement_sweeps = {measurements}
measure_every_sweeps = {interval}
[output]
campaign = "{campaign}"
run_id = "chain-{row:04d}"
root = "data/raw"
save_time_series = true
'''


def plan(args):
    checked_name(args.campaign)
    checked_name(args.release)
    if not 1 <= args.tasks <= 27:
        raise ValueError(
            "A batch must contain 1 to 27 tasks; reserve space for other work"
        )
    if not 1 <= args.minutes <= 1440:
        raise ValueError("Walltime must be between 1 and 1440 minutes")
    if not 1 <= args.replicas <= 16:
        raise ValueError("Replicas per start must be between 1 and 16")
    combinations = [
        (L, T, start, replica)
        for L in args.sizes
        for T in args.temperatures
        for start in args.starts
        for replica in range(args.replicas)
    ]
    if len(set(combinations)) != len(combinations) or len(combinations) > 1000:
        raise ValueError("Duplicate or excessive ensemble list")
    folder = REPO / "cluster" / "manifests" / args.campaign
    folder.mkdir(parents=True, exist_ok=False)
    rows = []
    for index, (L, T, start, replica) in enumerate(combinations):
        text = config_text(
            L,
            T,
            start,
            args.seed + index,
            args.campaign,
            index,
            args.adaptation,
            args.settling,
            args.measurements,
            args.interval,
        )
        launcher.validate(tomllib.loads(text))
        filename = f"chain-{index:04d}.toml"
        (folder / filename).write_text(text, encoding="utf-8", newline="\n")
        rows.append(
            {
                "config": filename,
                "sha256": launcher.sha256(folder / filename),
                "L": L,
                "T": T,
                "start": start,
                "seed": args.seed + index,
                "replica": replica,
            }
        )
    task_count = min(args.tasks, len(rows))
    manifest = {
        "schema_version": 1,
        "campaign": args.campaign,
        "release": args.release,
        "rows": rows,
        "tasks": task_count,
        "minutes": args.minutes,
        "maximum_cpu_hours": task_count * args.minutes / 60,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "partition": "requeue",
        "account": "general",
        "account_job_cap": 29,
        "method": "independent single-temperature sequential Metropolis; round-robin rows per worker",
    }
    (folder / "manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )
    print(
        json.dumps(
            {
                "manifest": str(folder / "manifest.json"),
                "chains": len(rows),
                "tasks": task_count,
                "maximum_cpu_hours": manifest["maximum_cpu_hours"],
            },
            indent=2,
        )
    )


def worker(manifest_path, task):
    path = Path(manifest_path).resolve()
    manifest = json.loads(path.read_text())
    if not 0 <= task < manifest["tasks"]:
        raise ValueError("Worker index out of bounds")
    binary = REPO / "build" / "mill-release" / "phi4"
    if launcher.sha256(binary) != manifest["binary_sha256"]:
        raise ValueError("Binary changed after submission")
    source = json.loads((REPO / "release.json").read_text())
    if source["release"] != manifest["release"]:
        raise ValueError("Wrong source release")
    for name, expected in source["files"].items():
        if launcher.sha256(REPO / name) != expected:
            raise ValueError(f"Source changed after deployment: {name}")
    for row in manifest["rows"][task :: manifest["tasks"]]:
        config = path.parent / row["config"]
        if launcher.sha256(config) != row["sha256"]:
            raise ValueError("Config changed after planning")
        launcher.run(config, binary)


def submit(args):
    path = Path(args.manifest).resolve()
    manifest = json.loads(path.read_text())
    maximum = manifest["tasks"] * manifest["minutes"] / 60
    if not 0 < args.budget_cpu_hours < float("inf") or maximum > args.budget_cpu_hours:
        raise ValueError(
            f"Batch maximum {maximum:.3f} CPU-hours exceeds the explicit submission budget"
        )
    if not 1 <= manifest["tasks"] <= 27 or not 1 <= manifest["minutes"] <= 1440:
        raise ValueError("Invalid task or walltime bounds")
    campaign = checked_name(manifest["campaign"])
    release = checked_name(manifest["release"])
    root = f"{REMOTE_BASE}/{release}"
    folder = f"{root}/cluster/manifests/{campaign}"
    if (path.parent / "submission.json").exists():
        raise ValueError("Already submitted; a retry requires a new campaign identity")
    for row in manifest["rows"]:
        if launcher.sha256(path.parent / row["config"]) != row["sha256"]:
            raise ValueError("Local manifest config hash mismatch")
    remote(f"set -eu; mkdir -p {root}/cluster/manifests; mkdir {folder}")
    subprocess.run(
        [
            "scp",
            "-o",
            "BatchMode=yes",
            str(path),
            *[str(path.parent / row["config"]) for row in manifest["rows"]],
            f"{DESTINATION}:{folder}/",
        ],
        check=True,
    )
    # Submission happens under one user lock. All existing array elements count.
    # Other clients submitting concurrently must also observe the user's cap.
    script = f"""set -euo pipefail
exec 9>/home/{USER}/phi4/submit.lock
flock -x 9
count=$(squeue -h -r -u {USER} -o '%i' | wc -l)
(( count + {manifest["tasks"]} <= 29 )) || {{ echo "Account job cap would be exceeded: $count existing" >&2; exit 2; }}
test -f {root}/build/mill-release/VALIDATED
python3.11 - {folder}/manifest.json {root}/build/mill-release/phi4 <<'PY'
import sys, json, hashlib
from pathlib import Path
path = Path(sys.argv[1])
data = json.loads(path.read_text())
data['binary_sha256'] = hashlib.sha256(Path(sys.argv[2]).read_bytes()).hexdigest()
path.write_text(json.dumps(data, indent=2) + '\\n')
PY
sbatch --parsable --partition=requeue --account=general --ntasks=1 --cpus-per-task=1 --mem=1G --time={manifest["minutes"]} --no-requeue --array=0-{manifest["tasks"] - 1} --job-name=phi4-{campaign} --output={root}/cluster/logs/%x-%A_%a.out --error={root}/cluster/logs/%x-%A_%a.err --wrap='cd {root} && export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 && python3.11 scripts/mill.py worker cluster/manifests/{campaign}/manifest.json "$SLURM_ARRAY_TASK_ID"'
"""
    record = {
        "manifest_sha256": launcher.sha256(path),
        "status": "SUBMITTING",
        "submitted_utc": datetime.now(timezone.utc).isoformat(),
    }
    record_path = path.parent / "submission.json"
    record_path.write_text(json.dumps(record, indent=2))
    try:
        result = remote("bash -s", input=script, text=True, capture_output=True)
        job = result.stdout.strip().splitlines()[-1]
        if not re.fullmatch(r"\d+(;[A-Za-z0-9_-]+)?", job):
            raise ValueError(f"Unrecognized sbatch output: {result.stdout}")
        record.update(status="SUBMITTED", job_id=job, remote_root=root)
    except Exception as error:
        record.update(status="SUBMISSION_UNCERTAIN", error=str(error))
        raise
    finally:
        record_path.write_text(json.dumps(record, indent=2) + "\n")
    print(json.dumps(record, indent=2))


def build(args):
    root = f"{REMOTE_BASE}/{checked_name(args.release)}"
    script = f"""set -euo pipefail
exec 9>/home/{USER}/phi4/submit.lock
flock -x 9
count=$(squeue -h -r -u {USER} -o '%i' | wc -l)
(( count < 29 )) || exit 2
test ! -e {root}/build/mill-release/phi4
mkdir {root}/cluster/build-submitted
sbatch --parsable --partition=requeue --account=general --ntasks=1 --cpus-per-task=1 --mem=2G --time=10 --no-requeue --job-name=phi4-build --output={root}/cluster/logs/build-%j.out --error={root}/cluster/logs/build-%j.err --wrap='set -eu; cd {root}; mkdir -p build/mill-release; g++ -std=c++17 -O3 -Wall -Wextra -Wpedantic -Iinclude -DPHI4_COMPILER=\\"GCC\\" -DPHI4_BUILD_TYPE=\\"Release\\" src/main.cpp -o build/mill-release/phi4; g++ -std=c++17 -O2 -Iinclude tests/cpp/test_model.cpp -o build/mill-release/test_model; g++ -std=c++17 -O2 -Iinclude tests/cpp/test_distribution.cpp -o build/mill-release/test_distribution; build/mill-release/test_model; build/mill-release/test_distribution; python3.11 tests/python/test_workflow.py build/mill-release/phi4; touch build/mill-release/VALIDATED'
"""
    remote("bash -s", input=script, text=True)


def pull(args):
    root = f"{REMOTE_BASE}/{checked_name(args.release)}"
    campaign = checked_name(args.campaign)
    target = (args.destination or REPO / "data" / "raw" / campaign).resolve()
    if not target.is_relative_to(REPO):
        raise ValueError("Pull destination must stay inside the repository")
    if target.exists():
        raise ValueError(
            "Local campaign already exists; use a new destination to preserve raw files"
        )
    target.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        [
            "scp",
            "-r",
            "-o",
            "BatchMode=yes",
            f"{DESTINATION}:{root}/data/raw/{campaign}",
            str(target),
        ],
        check=True,
    )
    print(target)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("deploy")
    sub.add_parser("status")
    for command in ("build", "pull"):
        p = sub.add_parser(command)
        p.add_argument("release")
        if command == "pull":
            p.add_argument("campaign")
            p.add_argument("--destination", type=Path)
    p = sub.add_parser("plan")
    p.add_argument("release")
    p.add_argument("campaign")
    p.add_argument("--sizes", nargs="+", type=int, required=True)
    p.add_argument("--temperatures", nargs="+", type=float, required=True)
    p.add_argument(
        "--starts",
        nargs="+",
        choices=["random", "ordered_positive", "ordered_negative"],
        default=["random", "ordered_positive"],
    )
    p.add_argument("--seed", type=int, default=540928000)
    p.add_argument("--tasks", type=int, default=12)
    p.add_argument("--replicas", type=int, default=1)
    p.add_argument("--minutes", type=int, default=10)
    p.add_argument("--adaptation", type=int, default=2000)
    p.add_argument("--settling", type=int, default=5000)
    p.add_argument("--measurements", type=int, default=20000)
    p.add_argument("--interval", type=int, default=1)
    p = sub.add_parser("submit")
    p.add_argument("manifest")
    p.add_argument("--budget-cpu-hours", type=float, required=True)
    p = sub.add_parser("worker")
    p.add_argument("manifest")
    p.add_argument("task", type=int)
    args = parser.parse_args()
    if args.command not in ("worker", "plan"):
        if not SITE_PATH.exists():
            parser.error(
                "Copy configs/mill.example.toml to configs/mill.local.toml and set your SSO ID"
            )
        if not re.fullmatch(r"[A-Za-z0-9_-]+", USER) or not re.fullmatch(
            r"[A-Za-z0-9.-]+", HOST
        ):
            parser.error("Invalid Mill host or user")
    if args.command == "status":
        remote(
            f"squeue -h -r -u {USER} -o '%i %T %j'; sacct -X -u {USER} --starttime today --format=JobID,JobName,State,Elapsed,AllocCPUS,TotalCPU -n"
        )
    elif args.command == "deploy":
        deploy()
    elif args.command == "worker":
        worker(args.manifest, args.task)
    else:
        globals()[args.command](args)


if __name__ == "__main__":
    main()
