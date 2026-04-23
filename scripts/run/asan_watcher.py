#!/usr/bin/env python3
"""ASAN watcher — decode corpus trees, run cuda-tile-opt (ASAN), save crashes.

Invoked from scripts/run/asan_watcher.sh, which sources env.sh and exports
the paths and tuning knobs this script needs.
"""

import hashlib
import json
import os
import re
import subprocess
import sys
import tempfile
import time
from pathlib import Path

from grammarinator.tool.tree_codec import FlatBuffersTreeCodec

PROJECT_ROOT = Path(os.environ["PROJECT_ROOT"])
BUILD_OUT = Path(os.environ["BUILD_OUT"])
CORPUS_DIR = Path(os.environ["CORPUS_DIR"])
CT_OPT_ASAN = os.environ["CT_OPT_ASAN"]

PERIOD = int(os.environ.get("WATCHER_PERIOD", "60"))
TIMEOUT = int(os.environ.get("WATCHER_TIMEOUT", "30"))
MAX_BYTES = int(os.environ.get("WATCHER_MAX_BYTES", "10485760"))

STATE_DIR = BUILD_OUT / "asan_watcher_state"
CRASH_DIR = BUILD_OUT / "asan_crashes"
LOG_FILE = BUILD_OUT / "asan_watcher.log"
SEEN_FILE = STATE_DIR / "seen.txt"
FINGERPRINT_FILE = STATE_DIR / "fingerprints.txt"

SANITIZER_LINE_RE = re.compile(r"^==\d+==\s*ERROR:\s*(AddressSanitizer|LeakSanitizer):\s*(.+)$")

CODEC = FlatBuffersTreeCodec()


def log(msg: str) -> None:
    line = f"{time.strftime('%Y-%m-%dT%H:%M:%S')} {msg}"
    print(line, flush=True)
    with LOG_FILE.open("a") as f:
        f.write(line + "\n")


def load_set(path: Path) -> set[str]:
    if not path.exists():
        return set()
    return {line.strip() for line in path.read_text().splitlines() if line.strip()}


def append_line(path: Path, line: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a") as f:
        f.write(line + "\n")


def iter_candidates() -> list[Path]:
    """Collect individual input files from corpus and workdir crashes dirs.

    Snapshots the current directory listing — files that disappear mid-scan
    are handled by later read failures (silently skipped).
    """
    candidates: list[Path] = []
    if CORPUS_DIR.is_dir():
        candidates.extend(p for p in CORPUS_DIR.iterdir() if p.is_file())
    for workdir in BUILD_OUT.glob("workdir_*"):
        for crashes_dir in workdir.glob("crashes.*"):
            if crashes_dir.is_dir():
                candidates.extend(p for p in crashes_dir.iterdir() if p.is_file())
    return candidates


def shadow_key(path: Path) -> str | None:
    """Identity tag for dedupe across loop iterations.

    mtime+size is cheap and stable; re-processing an input because Centipede
    rewrote it is fine — we dedupe crashes by ASAN fingerprint anyway.
    """
    try:
        st = path.stat()
    except OSError:
        return None
    return f"{path}|{st.st_mtime_ns}|{st.st_size}"


def decode_to_text(data: bytes) -> str | None:
    try:
        tree = CODEC.decode(data)
    except Exception:
        return None
    if tree is None:
        return None
    try:
        return str(tree)
    except Exception:
        return None


def extract_fingerprint(stderr: str) -> str | None:
    """First 'AddressSanitizer: <bug>' or 'LeakSanitizer: <bug>' header line.

    That's our bug-class key. Good enough for v1 — no stack hashing.
    """
    for line in stderr.splitlines():
        m = SANITIZER_LINE_RE.match(line.strip())
        if m:
            return f"{m.group(1)}: {m.group(2)}"
    return None


def run_asan(mlir_text: str) -> tuple[int, str]:
    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".mlir", delete=False, dir=STATE_DIR
    ) as tmp:
        tmp.write(mlir_text)
        tmp_path = tmp.name
    try:
        proc = subprocess.run(
            [CT_OPT_ASAN, "--allow-unregistered-dialect", tmp_path],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE,
            timeout=TIMEOUT,
        )
        return proc.returncode, proc.stderr.decode("utf-8", errors="replace")
    finally:
        try:
            os.unlink(tmp_path)
        except OSError:
            pass


def save_crash(mlir_text: str, stderr: str, fingerprint: str, returncode: int) -> str:
    short = hashlib.sha1(fingerprint.encode()).hexdigest()[:12]
    out = CRASH_DIR / short
    out.mkdir(parents=True, exist_ok=True)
    (out / "input.mlir").write_text(mlir_text)
    (out / "asan.log").write_text(stderr)
    header_lines = [
        line for line in stderr.splitlines()
        if line.strip()
    ][:5]
    meta = {
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "exit_code": returncode,
        "fingerprint": fingerprint,
        "header": header_lines,
    }
    (out / "meta.json").write_text(json.dumps(meta, indent=2))
    return short


def process_one(
    path: Path,
    fingerprints: set[str],
) -> tuple[bool, bool]:
    """Returns (processed, crashed_new)."""
    try:
        if path.stat().st_size > MAX_BYTES:
            return False, False
        data = path.read_bytes()
    except OSError:
        return False, False

    mlir_text = decode_to_text(data)
    if mlir_text is None:
        return False, False

    try:
        returncode, stderr = run_asan(mlir_text)
    except subprocess.TimeoutExpired:
        return True, False

    if returncode == 0:
        return True, False

    fingerprint = extract_fingerprint(stderr)
    if fingerprint is None:
        # Non-ASAN failure (regular cuda-tile rejection). Not our concern.
        return True, False

    if fingerprint in fingerprints:
        return True, False

    short = save_crash(mlir_text, stderr, fingerprint, returncode)
    fingerprints.add(fingerprint)
    append_line(FINGERPRINT_FILE, fingerprint)
    log(f"new_crash short={short} fingerprint={fingerprint!r}")
    return True, True


def run_batch(seen: set[str], fingerprints: set[str]) -> tuple[int, int]:
    processed = 0
    new_crashes = 0
    for path in iter_candidates():
        key = shadow_key(path)
        if key is None or key in seen:
            continue
        seen.add(key)
        append_line(SEEN_FILE, key)
        did_run, crashed_new = process_one(path, fingerprints)
        if did_run:
            processed += 1
        if crashed_new:
            new_crashes += 1
    return processed, new_crashes


def main() -> int:
    STATE_DIR.mkdir(parents=True, exist_ok=True)
    CRASH_DIR.mkdir(parents=True, exist_ok=True)

    seen = load_set(SEEN_FILE)
    fingerprints = load_set(FINGERPRINT_FILE)

    log(
        f"start period={PERIOD}s timeout={TIMEOUT}s "
        f"seen_loaded={len(seen)} fingerprints_loaded={len(fingerprints)}"
    )

    while True:
        t0 = time.time()
        try:
            processed, new_crashes = run_batch(seen, fingerprints)
        except Exception as e:
            log(f"batch_error {type(e).__name__}: {e}")
            processed, new_crashes = 0, 0
        elapsed = time.time() - t0
        log(
            f"batch processed={processed} new_crashes={new_crashes} "
            f"total_seen={len(seen)} elapsed={elapsed:.1f}s"
        )
        sleep_for = max(0, PERIOD - elapsed)
        time.sleep(sleep_for)


if __name__ == "__main__":
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        log("stop (SIGINT)")
        sys.exit(130)
