#!/usr/bin/env -S uv run python3
"""
check_coverage.py — Parse-coverage checker for the unified MLIR seed corpus.

The corpus at seeds/corpus/ contains tagged generic-form .mlir files named
<corpus>_<Dialect>__<n>.mlir, where <corpus> is 'cudatile' or 'upstream'.
This script parses them with the ANTLR grammar at resources/mlir.g4 and
reports pass rates by tag, clustered failure signatures, and per-tag timing.

Two modes that matter for the iterative-improvement loop:

  --regression-gate
      Require every cudatile_* file to parse. Exit non-zero if any fail.
      Independent of upstream pass rate. Prevents "fixed upstream by widening
      a rule that broke cuda-tile" regressions.

  --failure-report <path>
      Write a structured Markdown report of the run (per-tag rates, top
      failure clusters with example files). Format is stable so Claude Code
      or a human can diff reports across iterations.

Usage:
  ./scripts/check_coverage.py                           # default sample
  ./scripts/check_coverage.py --all                     # full corpus
  ./scripts/check_coverage.py --regression-gate --all   # for the iterative loop
  ./scripts/check_coverage.py --tags upstream_LLVMIR --all
  ./scripts/check_coverage.py --all --failure-report build/coverage_report.md
"""

from __future__ import annotations

import argparse
import importlib
import multiprocessing as mp
import os
import random
import re
import subprocess
import sys
import time
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[1]
GRAMMAR = PROJECT_ROOT / "resources" / "mlir.g4"
CORPUS_ROOT = PROJECT_ROOT / "seeds" / "corpus"
PARSER_CACHE = PROJECT_ROOT / "build" / "antlr"
START_RULE = "start"

# Representative sample for quick iteration; chosen to hit diverse failure
# modes quickly. Override with --tags.
DEFAULT_SAMPLE_TAGS = [
    "cudatile_CudaTile",
    "upstream_Arith",
    "upstream_Func",
    "upstream_SCF",
    "upstream_LLVMIR",
    "upstream_Vector",
    "upstream_Linalg",
    "upstream_MemRef",
    "upstream_Tosa",
    "upstream_Tensor",
    "upstream_SPIRV",
    "upstream_Affine",
]

# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument(
        "--corpus-root", type=Path, default=CORPUS_ROOT,
        help=f"Root of tagged .mlir corpus (default: {CORPUS_ROOT})",
    )
    p.add_argument(
        "--tags", nargs="+", default=None,
        help=f"Tags to include. Default: a representative sample.",
    )
    p.add_argument(
        "--per-tag", type=int, default=30,
        help="Max files per tag when sampling (default: 30)",
    )
    p.add_argument(
        "--all", action="store_true",
        help="Take every file for each tag (ignores --per-tag)",
    )
    p.add_argument(
        "--workers", type=int, default=max(1, (os.cpu_count() or 2) - 1),
        help="Parallel workers (default: CPU count - 1)",
    )
    p.add_argument(
        "--seed", type=int, default=0,
        help="RNG seed for reproducible sampling (default: 0)",
    )
    p.add_argument(
        "--no-sll", action="store_true",
        help="Skip SLL fast path; use LL-only (for benchmarking)",
    )
    p.add_argument(
        "--regression-gate", action="store_true",
        help="Exit non-zero if any cudatile_* file fails to parse",
    )
    p.add_argument(
        "--failure-report", type=Path, default=None,
        help="Write a structured Markdown failure report to this path",
    )
    p.add_argument(
        "--show-failures", type=int, default=5,
        help="Example files per failure cluster in terminal output (default: 5)",
    )
    p.add_argument(
        "-v", "--verbose", action="store_true",
        help="Print per-file pass/fail status",
    )
    return p.parse_args()


# ---------------------------------------------------------------------------
# Parser generation (cached)
# ---------------------------------------------------------------------------


def ensure_parser(grammar: Path, cache_dir: Path) -> None:
    """Regenerate ANTLR parser only when the grammar is newer than the cache."""
    cache_dir.mkdir(parents=True, exist_ok=True)
    grammar_name = grammar.stem
    lexer_py = cache_dir / f"{grammar_name}Lexer.py"
    parser_py = cache_dir / f"{grammar_name}Parser.py"

    needs_regen = (
        not lexer_py.exists()
        or not parser_py.exists()
        or grammar.stat().st_mtime > lexer_py.stat().st_mtime
    )
    if not needs_regen:
        return

    print("[parser] regenerating (grammar newer than cache)")
    cached_grammar = cache_dir / grammar.name
    cached_grammar.write_bytes(grammar.read_bytes())

    env = os.environ.copy()
    env.setdefault("ANTLR4_TOOLS_ANTLR_VERSION", "4.13.2")
    subprocess.run(
        ["antlr4", "-Dlanguage=Python3", "-visitor", cached_grammar.name],
        cwd=cache_dir, check=True, env=env,
    )


# ---------------------------------------------------------------------------
# Worker: parse one file
# ---------------------------------------------------------------------------


def _worker_init(cache_dir: str, grammar_name: str, no_sll: bool) -> None:
    global _Lexer, _Parser, _CommonTokenStream, _FileStream
    global _ErrorListener, _PredictionMode, _NO_SLL
    sys.path.insert(0, cache_dir)

    from antlr4 import CommonTokenStream, FileStream
    from antlr4.atn.PredictionMode import PredictionMode
    from antlr4.error.ErrorListener import ErrorListener as EL

    lexer_mod = importlib.import_module(f"{grammar_name}Lexer")
    parser_mod = importlib.import_module(f"{grammar_name}Parser")

    _Lexer = getattr(lexer_mod, f"{grammar_name}Lexer")
    _Parser = getattr(parser_mod, f"{grammar_name}Parser")
    _CommonTokenStream = CommonTokenStream
    _FileStream = FileStream
    _ErrorListener = EL
    _PredictionMode = PredictionMode
    _NO_SLL = no_sll


def _collecting_listener():
    errors: list[tuple[int, int, str]] = []

    class _EL(_ErrorListener):
        def syntaxError(self, recognizer, offendingSymbol, line, col, msg, e):
            errors.append((line, col, msg))

    return _EL(), errors


def _parse_file(path: str) -> tuple[str, bool, float, str]:
    """Returns (path, ok, elapsed, error_signature)."""
    t0 = time.perf_counter()
    try:
        stream = _FileStream(path, encoding="utf-8", errors="replace")
        lexer = _Lexer(stream)
        lexer.removeErrorListeners()
        lex_listener, lex_errors = _collecting_listener()
        lexer.addErrorListener(lex_listener)

        tokens = _CommonTokenStream(lexer)
        parser = _Parser(tokens)
        parser.removeErrorListeners()
        parse_listener, parse_errors = _collecting_listener()
        parser.addErrorListener(parse_listener)

        if not _NO_SLL:
            parser._interp.predictionMode = _PredictionMode.SLL
            try:
                parser.start()
            except Exception:
                tokens.reset()
                parser.reset()
                parser.removeErrorListeners()
                parse_errors.clear()
                lex_errors.clear()
                parser.addErrorListener(parse_listener)
                parser._interp.predictionMode = _PredictionMode.LL
                parser.start()
        else:
            parser._interp.predictionMode = _PredictionMode.LL
            parser.start()

        elapsed = time.perf_counter() - t0
        errs = lex_errors + parse_errors
        if errs:
            return (path, False, elapsed, _signature(errs[0]))
        return (path, True, elapsed, "")
    except Exception as e:
        elapsed = time.perf_counter() - t0
        return (path, False, elapsed, f"<exception: {type(e).__name__}>")


_TOKEN_RE = re.compile(r"'([^']+)'|<([^>]+)>")


def _signature(err: tuple[int, int, str]) -> str:
    """Short signature for clustering errors."""
    _, _, msg = err
    m = _TOKEN_RE.search(msg)
    if m:
        tok = m.group(1) or m.group(2)
        if "extraneous input" in msg:
            return f"extraneous '{tok}'"
        if "mismatched input" in msg:
            return f"mismatched '{tok}'"
        if "no viable alternative" in msg:
            return f"no viable alt at '{tok}'"
        if "missing" in msg:
            return f"missing '{tok}'"
        return f"at '{tok}'"
    return " ".join(msg.split()[:4])


# ---------------------------------------------------------------------------
# Sampling
# ---------------------------------------------------------------------------


def index_corpus(corpus_root: Path) -> dict[str, list[Path]]:
    """Return {tag: [paths]} for every file <tag>__<n>.mlir in the corpus."""
    by_tag: dict[str, list[Path]] = defaultdict(list)
    for f in corpus_root.glob("*.mlir"):
        if "__" not in f.name:
            continue
        tag = f.name.split("__", 1)[0]
        by_tag[tag].append(f)
    return by_tag


def select_files(
    by_tag: dict[str, list[Path]],
    tags: list[str] | None,
    per_tag: int,
    take_all: bool,
    seed: int,
) -> dict[str, list[Path]]:
    """Filter and sample the tags we want to parse."""
    rng = random.Random(seed)
    chosen_tags = tags if tags else DEFAULT_SAMPLE_TAGS
    result: dict[str, list[Path]] = {}
    for tag in chosen_tags:
        files = by_tag.get(tag, [])
        if not files:
            print(f"  [warn] no files with tag '{tag}'")
            result[tag] = []
            continue
        if take_all or len(files) <= per_tag:
            result[tag] = sorted(files)
        else:
            result[tag] = rng.sample(files, per_tag)
    return result


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------


def _pct(n: int, d: int) -> str:
    if d == 0:
        return "  -  "
    return f"{100.0 * n / d:5.1f}%"


def build_summary(
    results_by_tag: dict[str, list[tuple[str, bool, float, str]]],
) -> dict[str, Any]:
    """Compute the structured summary used by both terminal output and report."""
    tags_summary = {}
    total_n = total_pass = 0

    for tag, results in results_by_tag.items():
        n = len(results)
        passed = sum(1 for _, ok, _, _ in results if ok)
        total_n += n
        total_pass += passed

        if n == 0:
            tags_summary[tag] = {"n": 0, "pass": 0, "p50": None, "p99": None}
            continue

        times = sorted(t for _, _, t, _ in results)
        p50 = times[len(times) // 2]
        p99 = times[min(len(times) - 1, int(len(times) * 0.99))]
        tags_summary[tag] = {"n": n, "pass": passed, "p50": p50, "p99": p99}

    all_results = [r for results in results_by_tag.values() for r in results]
    failures = [(p, sig) for p, ok, _, sig in all_results if not ok]
    clusters: dict[str, list[str]] = defaultdict(list)
    for p, sig in failures:
        clusters[sig].append(p)

    return {
        "total_n": total_n,
        "total_pass": total_pass,
        "tags": tags_summary,
        "clusters": dict(clusters),
    }


def print_report(summary: dict[str, Any], show_failures: int, verbose: bool,
                 all_results: list[tuple[str, bool, float, str]]) -> None:
    """Terminal output."""
    print()
    print("Per-tag results:")
    print(f"  {'tag':<32} {'n':>5}  {'pass':>5}  {'rate':>6}  {'p50':>6}  {'p99':>7}")
    print(f"  {'-' * 32} {'-' * 5}  {'-' * 5}  {'-' * 6}  {'-' * 6}  {'-' * 7}")

    for tag, s in summary["tags"].items():
        if s["n"] == 0:
            print(f"  {tag:<32} {0:>5}  {0:>5}  {'  -  ':>6}  {'  -  ':>6}  {'  -  ':>7}")
            continue
        print(
            f"  {tag:<32} {s['n']:>5}  {s['pass']:>5}  "
            f"{_pct(s['pass'], s['n']):>6}  {s['p50']:>5.2f}s  {s['p99']:>6.2f}s"
        )

    print(f"  {'-' * 32} {'-' * 5}  {'-' * 5}  {'-' * 6}")
    print(
        f"  {'TOTAL':<32} {summary['total_n']:>5}  "
        f"{summary['total_pass']:>5}  {_pct(summary['total_pass'], summary['total_n']):>6}"
    )

    clusters = summary["clusters"]
    if clusters:
        n_fail = sum(len(v) for v in clusters.values())
        print()
        print(f"Failure clusters ({n_fail} failures, {len(clusters)} distinct signatures):")
        for sig, paths in sorted(clusters.items(), key=lambda kv: -len(kv[1])):
            print(f"  [{len(paths):>4}] {sig}")
            for p in paths[:show_failures]:
                print(f"         {Path(p).name}")
            if len(paths) > show_failures:
                print(f"         ... and {len(paths) - show_failures} more")

    if verbose:
        print()
        print("Per-file results:")
        for p, ok, t, sig in all_results:
            status = "PASS" if ok else f"FAIL [{sig}]"
            print(f"  {t:>6.2f}s  {status:<40} {Path(p).name}")


def write_failure_report(
    path: Path, summary: dict[str, Any], mode_label: str, elapsed: float,
) -> None:
    """Structured Markdown for Claude Code / humans between iterations.

    Format is stable: section headers, tables, and cluster bullets don't
    change order arbitrarily so diffs between iterations are meaningful.
    """
    total_n = summary["total_n"]
    total_pass = summary["total_pass"]
    clusters = summary["clusters"]

    lines: list[str] = []
    lines.append("# MLIR Grammar Coverage Report")
    lines.append("")
    lines.append(f"- Generated: {datetime.now().isoformat(timespec='seconds')}")
    lines.append(f"- Mode: {mode_label}")
    lines.append(f"- Runtime: {elapsed:.2f}s")
    lines.append(f"- Overall: **{total_pass} / {total_n} ({_pct(total_pass, total_n).strip()})**")
    lines.append("")

    # Regression-gate status at the top — most important signal for the loop.
    cudatile_tags = {t: s for t, s in summary["tags"].items() if t.startswith("cudatile_")}
    if cudatile_tags:
        ct_n = sum(s["n"] for s in cudatile_tags.values())
        ct_pass = sum(s["pass"] for s in cudatile_tags.values())
        status = "PASS" if ct_n == ct_pass else "FAIL"
        lines.append(f"## Regression gate (cuda-tile): **{status}** ({ct_pass} / {ct_n})")
        lines.append("")
        if ct_n != ct_pass:
            lines.append("Cuda-tile regressions:")
            for tag, s in cudatile_tags.items():
                if s["n"] != s["pass"]:
                    lines.append(f"- `{tag}`: {s['pass']}/{s['n']}")
            lines.append("")

    # Per-tag table.
    lines.append("## Per-tag results")
    lines.append("")
    lines.append("| Tag | Files | Pass | Rate | p50 | p99 |")
    lines.append("|---|---:|---:|---:|---:|---:|")
    for tag, s in sorted(summary["tags"].items()):
        if s["n"] == 0:
            lines.append(f"| `{tag}` | 0 | 0 | — | — | — |")
            continue
        lines.append(
            f"| `{tag}` | {s['n']} | {s['pass']} | "
            f"{_pct(s['pass'], s['n']).strip()} | {s['p50']:.2f}s | {s['p99']:.2f}s |"
        )
    lines.append("")

    # Failure clusters, largest first.
    if clusters:
        lines.append("## Failure clusters")
        lines.append("")
        lines.append("Clusters are ordered by failure count. Fixing the top cluster is almost always the highest-leverage edit.")
        lines.append("")
        for sig, paths in sorted(clusters.items(), key=lambda kv: -len(kv[1])):
            lines.append(f"### `{sig}` — {len(paths)} failures")
            lines.append("")
            tag_counts: dict[str, int] = defaultdict(int)
            for p in paths:
                name = Path(p).name
                if "__" in name:
                    tag_counts[name.split("__", 1)[0]] += 1
            if tag_counts:
                tag_str = ", ".join(f"`{t}` ({c})" for t, c in sorted(tag_counts.items(), key=lambda kv: -kv[1]))
                lines.append(f"Tags affected: {tag_str}")
                lines.append("")
            lines.append("Example files:")
            for p in paths[:10]:
                lines.append(f"- `{Path(p).name}`")
            if len(paths) > 10:
                lines.append(f"- … and {len(paths) - 10} more")
            lines.append("")
    else:
        lines.append("## No failures")
        lines.append("")

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n")
    print(f"[report] wrote {path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> int:
    args = parse_args()

    if not args.corpus_root.exists():
        print(f"ERROR: corpus root not found: {args.corpus_root}", file=sys.stderr)
        print("  Run scripts/build/organize_seeds.sh first.", file=sys.stderr)
        return 2

    ensure_parser(GRAMMAR, PARSER_CACHE)
    by_tag = index_corpus(args.corpus_root)

    # Regression gate always parses all cudatile_* files regardless of --tags.
    if args.regression_gate:
        cudatile_tags = [t for t in by_tag if t.startswith("cudatile_")]
        selected = select_files(
            by_tag,
            sorted(set((args.tags or DEFAULT_SAMPLE_TAGS) + cudatile_tags)),
            args.per_tag,
            args.all or True,  # always all for cudatile
            args.seed,
        )
        # Downgrade the non-cudatile selections to honor --per-tag / --all.
        non_ct = select_files(
            by_tag,
            args.tags if args.tags else DEFAULT_SAMPLE_TAGS,
            args.per_tag, args.all, args.seed,
        )
        for t in non_ct:
            if not t.startswith("cudatile_"):
                selected[t] = non_ct[t]
    else:
        selected = select_files(
            by_tag,
            args.tags if args.tags else DEFAULT_SAMPLE_TAGS,
            args.per_tag, args.all, args.seed,
        )

    work: list[tuple[str, Path]] = [
        (tag, p) for tag, ps in selected.items() for p in ps
    ]
    total_files = len(work)
    tags_used = len([t for t, ps in selected.items() if ps])
    mode_label = "all files" if args.all else f"sample of up to {args.per_tag} per tag"
    if args.regression_gate:
        mode_label += " (regression gate on)"

    print(f"[corpus] {total_files} files across {tags_used} tags ({mode_label})")
    if args.workers > 1:
        print(f"[parse] {args.workers} workers")

    t_start = time.perf_counter()
    results_by_tag: dict[str, list[tuple[str, bool, float, str]]] = defaultdict(list)

    if args.workers == 1:
        _worker_init(str(PARSER_CACHE), GRAMMAR.stem, args.no_sll)
        for tag, path in work:
            results_by_tag[tag].append(_parse_file(str(path)))
    else:
        paths_only = [str(p) for _, p in work]
        path_to_tag = {str(p): tag for tag, p in work}
        with mp.Pool(
            args.workers,
            initializer=_worker_init,
            initargs=(str(PARSER_CACHE), GRAMMAR.stem, args.no_sll),
        ) as pool:
            for result in pool.imap_unordered(_parse_file, paths_only, chunksize=8):
                results_by_tag[path_to_tag[result[0]]].append(result)

    elapsed = time.perf_counter() - t_start
    print(f"[parse] done in {elapsed:.2f}s")

    summary = build_summary(dict(results_by_tag))
    all_results = [r for rs in results_by_tag.values() for r in rs]
    print_report(summary, args.show_failures, args.verbose, all_results)

    if args.failure_report:
        write_failure_report(args.failure_report, summary, mode_label, elapsed)

    # Exit codes:
    #   0 — regression gate OK (if enabled) and all parsed
    #   1 — at least one failure in the selected set
    #   3 — regression gate failed (cuda-tile broken)
    if args.regression_gate:
        ct_n = sum(len(rs) for t, rs in results_by_tag.items() if t.startswith("cudatile_"))
        ct_pass = sum(
            sum(1 for _, ok, _, _ in rs if ok)
            for t, rs in results_by_tag.items()
            if t.startswith("cudatile_")
        )
        if ct_n != ct_pass:
            print(f"[gate] FAIL: cudatile {ct_pass}/{ct_n}", file=sys.stderr)
            return 3

    return 0 if summary["total_pass"] == summary["total_n"] else 1


if __name__ == "__main__":
    sys.exit(main())