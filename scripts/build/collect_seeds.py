import multiprocessing
import subprocess
from pathlib import Path
import shutil
import os
import re
from tqdm import tqdm
import click
import logging

logger = logging.getLogger(__name__)

NEGATIVE_DIRECTIVE_RE = re.compile(r"expected-(error|note|remark|warning)")


@click.command()
@click.argument(
    "repo_dir",
    type=click.Path(exists=True, file_okay=False, path_type=Path),
)
@click.argument(
    "output_dir",
    type=click.Path(path_type=Path),
)
@click.option(
    "--mlir-opt-path",
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
    required=True,
)
@click.option("--workers", default=os.cpu_count())
@click.option("--log-level", default="INFO")
def main(
    repo_dir: Path, output_dir: Path, mlir_opt_path: Path, workers: int, log_level: str
):
    logger.setLevel(getattr(logging, log_level))
    logger.addHandler(logging.StreamHandler())

    original_dir = output_dir / "original"
    split_dir = output_dir / "split"
    generic_dir = output_dir / "generic"
    fail_log = output_dir / "conversion_failures.log"

    for d in [original_dir, split_dir, generic_dir]:
        if d.exists():
            shutil.rmtree(d)
        d.mkdir(parents=True)
    if fail_log.exists():
        fail_log.unlink()

    find_and_copy_mlir(repo_dir, original_dir)
    split_mlir_tests(original_dir, split_dir)
    convert_all_to_generic(split_dir, generic_dir, mlir_opt_path, workers, fail_log)

    logger.info(f"\nDone. Generic MLIR seeds in:\n{generic_dir}")


# --------------------------------------------------
# Step 1: Collect MLIR files
# --------------------------------------------------
#
# Each file is copied with a path-derived tag so the dialect it came from
# survives through splitting and conversion. A file originally at
#     mlir/test/Dialect/LLVMIR/invalid.mlir
# becomes
#     original/LLVMIR__invalid.mlir
# which keeps per-dialect breakdowns possible at the end of the pipeline.
#
# Files whose path or contents mark them as negative tests (meant to fail
# parsing or verification) are skipped — they'd just fail conversion in
# predictable ways and pollute the failure signal.


def dialect_tag_for(path: Path, repo_dir: Path) -> str:
    """Extract a dialect-ish tag from the file's path under the repo.

    Typical upstream layouts:
        mlir/test/Dialect/<Dialect>/...       -> <Dialect>
        mlir/test/Conversion/<Name>/...       -> conv_<Name>
        mlir/test/Transforms/...              -> transforms
        anywhere else                         -> use immediate parent dir name
    """
    try:
        rel = path.relative_to(repo_dir).parts
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
    """Heuristic: skip files that are meant to fail parsing/verification."""
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


def find_and_copy_mlir(repo_dir: Path, dst_dir: Path):
    logger.info("Collecting MLIR files...")

    files = list(repo_dir.rglob("*.mlir"))
    skipped = 0

    for file in tqdm(files):
        if is_negative_test(file):
            skipped += 1
            continue

        tag = dialect_tag_for(file, repo_dir)
        dst = dst_dir / f"{tag}__{file.name}"
        i = 1
        while dst.exists():
            dst = dst_dir / f"{tag}__{file.stem}_{i}.mlir"
            i += 1

        shutil.copy(file, dst)

    copied = len(list(dst_dir.glob("*.mlir")))
    logger.info(f"  copied {copied} files, skipped {skipped} negative tests")


# --------------------------------------------------
# Step 2: Split LLVM test files on `// -----`
# --------------------------------------------------
#
# lit-style tests often pack multiple independent IR cases into one file,
# separated by `// -----`. Splitting each file into its partitions means
# a bad partition can't take the good ones down with it.
#
# All comment lines are stripped. Tag prefix is preserved so dialect
# provenance survives.


def split_mlir_tests(src_dir: Path, dst_dir: Path):
    logger.info("Splitting MLIR tests...")

    files = list(src_dir.glob("*.mlir"))
    total_partitions = 0

    for file in tqdm(files):
        cases = []
        current = ""

        with open(file) as f:
            for line in f:
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

            out = dst_dir / f"{file.stem}_{i}.mlir"
            with open(out, "w") as f:
                f.write(case)
            total_partitions += 1

    logger.info(f"  produced {total_partitions} partitions from {len(files)} files")


# --------------------------------------------------
# Step 3: Convert to generic MLIR
# --------------------------------------------------
#
# Round-trip through mlir-opt with --mlir-print-op-generic. On failure we
# capture the first few lines of stderr so the aggregated failure log tells
# us *why* conversion failed, not just that it did.
#
# convert_one returns (input_path, ok, err_snippet) so the driver can both
# count outcomes and assemble the failure log without additional I/O.


def convert_one(args):
    input_file, output_file, mlir_opt = args

    try:
        result = subprocess.run(
            [
                str(mlir_opt),
                "--allow-unregistered-dialect",
                "--mlir-print-op-generic",
                "--mlir-print-debuginfo=false",
                str(input_file),
            ],
            capture_output=True,
            text=True,
        )

        if result.returncode != 0:
            err_head = "\n".join(result.stderr.splitlines()[:5])
            return (str(input_file), False, err_head)

        if not result.stdout.strip():
            return (str(input_file), False, "<empty stdout>")

        with open(output_file, "w") as f:
            f.write(result.stdout)

        return (str(input_file), True, "")

    except Exception as e:
        return (str(input_file), False, f"<exception: {e}>")


def convert_all_to_generic(
    src_dir: Path, dst_dir: Path, mlir_opt: Path, workers: int, fail_log: Path
):
    logger.info("Converting to generic MLIR...")

    inputs = list(src_dir.glob("*.mlir"))
    args = [(inp, dst_dir / inp.name, mlir_opt) for inp in inputs]

    succeeded = 0
    failed = 0

    with multiprocessing.Pool(workers) as pool, fail_log.open("w") as flog:
        for input_path, ok, err in tqdm(
            pool.imap_unordered(convert_one, args),
            total=len(args),
        ):
            if ok:
                succeeded += 1
            else:
                failed += 1
                flog.write(f"=== {input_path}\n{err}\n\n")

    logger.info(
        f"  converted {succeeded}/{len(args)} partitions "
        f"({failed} failures logged to {fail_log})"
    )

    # Per-dialect breakdown — filename prefix before `__` is the tag.
    counts: dict[str, int] = {}
    for p in dst_dir.glob("*.mlir"):
        tag = p.name.split("__", 1)[0] if "__" in p.name else "other"
        counts[tag] = counts.get(tag, 0) + 1
    if counts:
        logger.info("  per-dialect output counts:")
        for tag in sorted(counts):
            logger.info(f"    {tag:<16} {counts[tag]}")


if __name__ == "__main__":
    main()