#!/usr/bin/env python3

from __future__ import annotations

import multiprocessing
import os
import re
import shutil
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

from tqdm import tqdm

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SEEDS_DIR = PROJECT_ROOT / "seeds"
DEPS_DIR = PROJECT_ROOT / "deps"

# (source_path, tag_prefix, opt_binary_path)
# tag_prefix becomes the filename prefix: cudatile_CudaTile__foo.mlir etc.
# Adjust paths here if layout changes — this is the one place.
SOURCES: list[tuple[Path, str, Path]] = [
    (
        DEPS_DIR / "cuda-tile" / "test",
        "cudatile",
        DEPS_DIR / "cuda-tile-build-sancov" / "bin" / "cuda-tile-opt",
    ),
    (
        DEPS_DIR / "llvm-project" / "mlir" / "test",
        "upstream",
        DEPS_DIR / "llvm-build-plain" / "bin" / "mlir-opt",
    ),
]

NEGATIVE_DIRECTIVE_RE = re.compile(r"expected-(error|note|remark|warning)")


@dataclass
class ConversionResult:
    path: str
    ok: bool
    err: str


# ---------------------------------------------------------------------------
# Path tagging
# ---------------------------------------------------------------------------


def dialect_tag_for(path: Path, repo_root: Path) -> str:
    """Extract a dialect tag from the file's path under the repo root.

    Handles the typical MLIR test layouts (Dialect/<X>, Conversion/<Y>,
    Transforms/). Falls back to immediate parent dir name.
    """
    try:
        rel = path.relative_to(repo_root).parts
    except ValueError:
        return path.parent.name

    if "Dialect" in rel:
        i = rel.index("Dialect")
        if i + 1 < len(rel):
            return rel[i + 1]
    if "Conversion" in rel:
        i = rel.index("Conversion")
        if i + 1 < len(rel):
            return f"conv_{rel[i + 1]}"
    if "Transforms" in rel:
        return "transforms"
    return path.parent.name


def is_negative_test(path: Path) -> bool:
    if "invalid" in path.name.lower():
        return True
    try:
        with path.open("r", errors="replace") as f:
            for line in f:
                if NEGATIVE_DIRECTIVE_RE.search(line):
                    return True
    except OSError:
        return True
    return False


# ---------------------------------------------------------------------------
# Step 1: copy
# ---------------------------------------------------------------------------


def copy_source(repo_root: Path, tag_prefix: str, original_dir: Path) -> int:
    files = list(repo_root.rglob("*.mlir"))
    copied = 0
    skipped = 0
    for f in tqdm(files, desc=f"  {tag_prefix}"):
        if is_negative_test(f):
            skipped += 1
            continue
        tag = dialect_tag_for(f, repo_root)
        dst = original_dir / f"{tag_prefix}_{tag}__{f.name}"
        i = 1
        while dst.exists():
            dst = original_dir / f"{tag_prefix}_{tag}__{f.stem}_{i}.mlir"
            i += 1
        shutil.copy(f, dst)
        copied += 1
    print(f"  [{tag_prefix}] copied {copied}, skipped {skipped} negative tests")
    return copied


# ---------------------------------------------------------------------------
# Step 2: split on `// -----`
# ---------------------------------------------------------------------------


def split_all(original_dir: Path, split_dir: Path) -> int:
    files = list(original_dir.glob("*.mlir"))
    total = 0
    for f in tqdm(files, desc="  splitting"):
        cases = []
        current = ""
        with open(f) as fh:
            for line in fh:
                if line.startswith("// -----"):
                    cases.append(current)
                    current = ""
                    continue
                if line.strip().startswith("//"):
                    continue
                current += line
        if current.strip():
            cases.append(current)

        for i, case in enumerate(cases):
            if not case.strip():
                continue
            out = split_dir / f"{f.stem}_{i}.mlir"
            out.write_text(case)
            total += 1
    print(f"  produced {total} partitions from {len(files)} files")
    return total


# ---------------------------------------------------------------------------
# Step 3: round-trip through mlir-opt
# ---------------------------------------------------------------------------
#
# Each source's files are handled with its own opt binary. We figure out
# which binary to use by looking at the filename's corpus prefix.


def _convert_one(args: tuple[Path, Path, dict[str, str]]) -> ConversionResult:
    input_file, output_file, tag_to_opt = args
    corpus_tag = input_file.name.split("_", 1)[0]
    opt_binary = tag_to_opt.get(corpus_tag)
    if opt_binary is None:
        return ConversionResult(str(input_file), False, f"<no opt for tag '{corpus_tag}'>")

    try:
        result = subprocess.run(
            [
                opt_binary,
                "--allow-unregistered-dialect",
                "--mlir-print-op-generic",
                "--mlir-print-debuginfo=false",
                str(input_file),
            ],
            capture_output=True,
            text=True,
            timeout=30,
        )
        if result.returncode != 0:
            return ConversionResult(
                str(input_file), False, "\n".join(result.stderr.splitlines()[:5])
            )
        if not result.stdout.strip():
            return ConversionResult(str(input_file), False, "<empty stdout>")
        output_file.write_text(result.stdout)
        return ConversionResult(str(input_file), True, "")
    except subprocess.TimeoutExpired:
        return ConversionResult(str(input_file), False, "<timeout>")
    except Exception as e:
        return ConversionResult(str(input_file), False, f"<exception: {e}>")


def convert_all(
    split_dir: Path,
    generic_dir: Path,
    tag_to_opt: dict[str, str],
    workers: int,
    fail_log: Path,
) -> tuple[int, int]:
    inputs = list(split_dir.glob("*.mlir"))
    args = [(inp, generic_dir / inp.name, tag_to_opt) for inp in inputs]

    succeeded = 0
    failed = 0
    with multiprocessing.Pool(workers) as pool, fail_log.open("w") as flog:
        for r in tqdm(
            pool.imap_unordered(_convert_one, args, chunksize=16),
            total=len(args),
            desc="  converting",
        ):
            if r.ok:
                succeeded += 1
            else:
                failed += 1
                flog.write(f"=== {r.path}\n{r.err}\n\n")

    return succeeded, failed


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def per_tag_counts(generic_dir: Path) -> dict[str, int]:
    counts: dict[str, int] = {}
    for p in generic_dir.glob("*.mlir"):
        tag = p.name.split("__", 1)[0] if "__" in p.name else "other"
        counts[tag] = counts.get(tag, 0) + 1
    return counts


def main() -> int:
    # Validate sources and opt binaries before doing any work.
    tag_to_opt: dict[str, str] = {}
    for repo, tag, opt in SOURCES:
        if not repo.exists():
            print(f"ERROR: source repo not found: {repo}", file=sys.stderr)
            return 1
        if not opt.exists():
            print(f"ERROR: opt binary not found: {opt}", file=sys.stderr)
            return 1
        tag_to_opt[tag] = str(opt)

    original_dir = SEEDS_DIR / "original"
    split_dir = SEEDS_DIR / "split"
    generic_dir = SEEDS_DIR / "generic"
    fail_log = SEEDS_DIR / "conversion_failures.log"

    # Wipe and recreate; seed collection is idempotent by construction.
    for d in (original_dir, split_dir, generic_dir):
        if d.exists():
            shutil.rmtree(d)
        d.mkdir(parents=True)
    if fail_log.exists():
        fail_log.unlink()

    workers = os.cpu_count() or 4

    print("Collecting MLIR files...")
    for repo, tag, _ in SOURCES:
        copy_source(repo, tag, original_dir)

    print("\nSplitting...")
    split_all(original_dir, split_dir)

    print("\nConverting to generic...")
    succeeded, failed = convert_all(
        split_dir, generic_dir, tag_to_opt, workers, fail_log
    )
    print(f"  converted {succeeded} partitions, {failed} failures logged to {fail_log}")

    # split/ is intermediate; remove to keep seeds/ tidy.
    shutil.rmtree(split_dir)

    total = len(list(generic_dir.glob("*.mlir")))
    print(f"\n[done] {total} generic-form files in {generic_dir}")

    counts = per_tag_counts(generic_dir)
    if counts:
        print(f"\nPer-tag counts ({len(counts)} tags):")
        for tag, n in sorted(counts.items(), key=lambda kv: -kv[1]):
            print(f"  {tag:<40} {n}")

    return 0


if __name__ == "__main__":
    sys.exit(main())