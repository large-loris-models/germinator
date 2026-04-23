#!/usr/bin/env python3
"""Status snapshot of Centipede + ASAN watcher + corpus state.

Read-only. Sourced from scripts/run/status.sh (which sets up env).
"""

import csv
import json
import os
import re
import subprocess
import time
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

PROJECT_ROOT = Path(os.environ["PROJECT_ROOT"])
BUILD_OUT = Path(os.environ["BUILD_OUT"])
CORPUS_DIR = Path(os.environ["CORPUS_DIR"])
SEEDS_DIR = Path(os.environ["SEEDS_DIR"])

WATCHER_PERIOD = int(os.environ.get("WATCHER_PERIOD", "60"))

ASAN_CRASHES = BUILD_OUT / "asan_crashes"
WATCHER_LOG = BUILD_OUT / "asan_watcher.log"


def fmt_duration(seconds: float) -> str:
    if seconds < 0:
        return "n/a"
    s = int(seconds)
    d, s = divmod(s, 86400)
    h, s = divmod(s, 3600)
    m, s = divmod(s, 60)
    if d:
        return f"{d}d{h:02d}h{m:02d}m"
    if h:
        return f"{h}h{m:02d}m{s:02d}s"
    if m:
        return f"{m}m{s:02d}s"
    return f"{s}s"


def fmt_mtime(ts: float) -> str:
    return datetime.fromtimestamp(ts).strftime("%Y-%m-%d %H:%M:%S")


def newest_workdir() -> Path | None:
    workdirs = sorted(BUILD_OUT.glob("workdir_*"), key=lambda p: p.stat().st_mtime)
    return workdirs[-1] if workdirs else None


def find_centipede_pids() -> list[tuple[int, float]]:
    """Return (pid, start_epoch) for running centipede processes."""
    try:
        out = subprocess.run(
            ["pgrep", "-a", "centipede"],
            capture_output=True, text=True, check=False,
        )
    except FileNotFoundError:
        return []
    pids: list[tuple[int, float]] = []
    for line in out.stdout.splitlines():
        parts = line.split(None, 1)
        if not parts:
            continue
        try:
            pid = int(parts[0])
        except ValueError:
            continue
        try:
            proc_stat = Path(f"/proc/{pid}").stat()
        except OSError:
            continue
        pids.append((pid, proc_stat.st_ctime))
    return pids


def find_watcher_pid() -> int | None:
    try:
        out = subprocess.run(
            ["pgrep", "-f", "asan_watcher.py"],
            capture_output=True, text=True, check=False,
        )
    except FileNotFoundError:
        return None
    for line in out.stdout.splitlines():
        try:
            return int(line.strip())
        except ValueError:
            continue
    return None


def centipede_coverage(workdir: Path) -> str:
    """Last known PC count from the fuzzing-stats CSV."""
    csvs = sorted(workdir.glob("fuzzing-stats-*.csv"))
    if not csvs:
        return "unknown"
    last = csvs[-1]
    try:
        with last.open() as f:
            reader = csv.DictReader(f)
            rows = list(reader)
    except OSError:
        return "unknown"
    if not rows:
        return "unknown"
    last_row = rows[-1]
    key = "NumCoveredPcs_Max"
    if key in last_row and last_row[key]:
        return last_row[key]
    return "unknown"


CRASH_SIG_RE = re.compile(r"^\s*(\S.*?)\s*$")


def centipede_crash_bucketing(workdir: Path) -> tuple[int, Counter]:
    """Count crash inputs and bucket by first line of crash-metadata/*.desc."""
    total = 0
    buckets: Counter = Counter()
    for crashes_dir in workdir.glob("crashes.*"):
        if crashes_dir.is_dir():
            total += sum(1 for _ in crashes_dir.iterdir())
    for meta_dir in workdir.glob("crash-metadata.*"):
        if not meta_dir.is_dir():
            continue
        for desc in meta_dir.glob("*.desc"):
            try:
                first = desc.read_text(errors="replace").splitlines()
            except OSError:
                continue
            label = first[0].strip() if first else "(empty)"
            buckets[label] += 1
    return total, buckets


def last_log_line(path: Path) -> tuple[str | None, float | None]:
    if not path.exists():
        return None, None
    try:
        text = path.read_text()
    except OSError:
        return None, None
    lines = [ln for ln in text.splitlines() if ln.strip()]
    if not lines:
        return None, None
    last = lines[-1]
    m = re.match(r"^(\S+)\s", last)
    if not m:
        return last, None
    try:
        ts = datetime.strptime(m.group(1), "%Y-%m-%dT%H:%M:%S").timestamp()
    except ValueError:
        return last, None
    return last, ts


def watcher_totals() -> tuple[int, str | None]:
    """Returns (processed_total, last_batch_line)."""
    if not WATCHER_LOG.exists():
        return 0, None
    total = 0
    last_batch = None
    try:
        for line in WATCHER_LOG.read_text().splitlines():
            m = re.search(r"\bprocessed=(\d+)\b", line)
            if m and "batch" in line:
                total += int(m.group(1))
                last_batch = line
    except OSError:
        return 0, None
    return total, last_batch


def count_dir(path: Path) -> int:
    if not path.is_dir():
        return 0
    return sum(1 for p in path.iterdir() if p.is_file())


def count_subdirs(path: Path) -> int:
    if not path.is_dir():
        return 0
    return sum(1 for p in path.iterdir() if p.is_dir())


def recent_crashes(path: Path, within: float) -> int:
    if not path.is_dir():
        return 0
    cutoff = time.time() - within
    return sum(
        1 for p in path.iterdir()
        if p.is_dir() and p.stat().st_mtime >= cutoff
    )


def age_range(paths: list[Path]) -> str:
    if not paths:
        return "n/a"
    mtimes = [p.stat().st_mtime for p in paths]
    return f"{fmt_mtime(min(mtimes))} → {fmt_mtime(max(mtimes))}"


def main() -> None:
    now = time.time()
    print(f"=== Fuzzer Status === {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print()

    # ─ Centipede ──────────────────────────────────────────────────────────
    workdir = newest_workdir()
    pids = find_centipede_pids()
    print("Centipede:")
    if workdir is None:
        print("  workdir:       (none)")
    else:
        print(f"  workdir:       {workdir}")

    inputs_total = count_dir(CORPUS_DIR)
    print(f"  inputs total:  {inputs_total}")

    if workdir is not None:
        crash_total, buckets = centipede_crash_bucketing(workdir)
        if not buckets:
            print(f"  crashes:       {crash_total}")
        else:
            print(f"  crashes:       {crash_total}")
            for label, n in buckets.most_common(10):
                trimmed = label if len(label) <= 72 else label[:69] + "..."
                print(f"    {n:>4}  {trimmed}")
        coverage = centipede_coverage(workdir)
        print(f"  coverage:      {coverage}")
    else:
        print("  crashes:       0")
        print("  coverage:      unknown")

    if pids:
        oldest = min(start for _, start in pids)
        print(f"  uptime:        {fmt_duration(now - oldest)} ({len(pids)} proc)")
        print("  status:        RUNNING")
    else:
        # No process; decide STOPPED vs CRASHED from recent workdir activity.
        status = "STOPPED"
        if workdir is not None:
            latest = max(
                (p.stat().st_mtime for p in workdir.rglob("*") if p.is_file()),
                default=0,
            )
            if latest and (now - latest) < 600:
                status = "CRASHED (no process, recent activity)"
        print("  uptime:        n/a")
        print(f"  status:        {status}")

    print()

    # ─ ASAN watcher ───────────────────────────────────────────────────────
    print("ASAN watcher:")
    watcher_pid = find_watcher_pid()
    processed_total, last_batch = watcher_totals()
    last_line, last_ts = last_log_line(WATCHER_LOG)

    if last_ts is not None:
        age = now - last_ts
        print(f"  last log:      {fmt_duration(age)} ago")
    else:
        print("  last log:      n/a")

    print(f"  processed:     {processed_total}")

    new_24h = recent_crashes(ASAN_CRASHES, within=86400)
    total_crashes = count_subdirs(ASAN_CRASHES)
    print(f"  new crashes:   {new_24h} (last 24h)")
    print(f"  total crashes: {total_crashes}")

    if watcher_pid is not None:
        try:
            start = Path(f"/proc/{watcher_pid}").stat().st_ctime
            print(f"  uptime:        {fmt_duration(now - start)} (pid {watcher_pid})")
        except OSError:
            print(f"  uptime:        n/a (pid {watcher_pid})")
        if last_ts is not None and (now - last_ts) < 2 * WATCHER_PERIOD:
            status = "RUNNING"
        else:
            status = "IDLE"
        print(f"  status:        {status}")
    else:
        if last_ts is None:
            print("  status:        STOPPED (no log)")
        elif (now - last_ts) < 600:
            print(f"  status:        CRASHED (see {WATCHER_LOG})")
        else:
            print("  status:        STOPPED")

    print()

    # ─ Corpus ─────────────────────────────────────────────────────────────
    print("Corpus:")
    trees_dir = SEEDS_DIR / "trees"
    seed_count = sum(1 for _ in trees_dir.glob("*.grtf")) if trees_dir.is_dir() else 0
    print(f"  seeds:         {seed_count}")

    centipede_inputs = count_dir(CORPUS_DIR)
    shard_bytes = 0
    if workdir is not None:
        for shard in workdir.glob("corpus.*"):
            try:
                shard_bytes += shard.stat().st_size
            except OSError:
                pass
    print(f"  centipede:     {centipede_inputs} files in corpus/  "
          f"({shard_bytes / (1024 * 1024):.1f} MB across workdir shards)")

    all_inputs = list(CORPUS_DIR.iterdir()) if CORPUS_DIR.is_dir() else []
    all_inputs = [p for p in all_inputs if p.is_file()]
    print(f"  age range:     {age_range(all_inputs)}")


if __name__ == "__main__":
    main()
