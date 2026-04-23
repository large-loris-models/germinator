#!/usr/bin/env -S uv run python3
"""Dump an ANTLR parse tree for a seed file as `toStringTree` text.

Useful artifact for mutator work — lets you see what rules are tagged where.

Usage:
    scripts/analysis/dump_parse_tree.py [seed.mlir] [out.txt]

Defaults to attrsTest.mlir (representative: has ops, regions, prop dicts,
multi-dimensional types, unit attributes, nested dialect-type bodies with
key=value pairs) and build/parse_tree_sample.txt.
"""
from __future__ import annotations

import importlib
import os
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
GRAMMAR = REPO / "resources" / "mlir.g4"
PARSER_CACHE = REPO / "build" / "antlr"


def ensure_parser(grammar: Path = GRAMMAR, cache_dir: Path = PARSER_CACHE) -> None:
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


def load_parser(cache_dir: Path = PARSER_CACHE, grammar_name: str = GRAMMAR.stem):
    sys.path.insert(0, str(cache_dir))
    lexer_mod = importlib.import_module(f"{grammar_name}Lexer")
    parser_mod = importlib.import_module(f"{grammar_name}Parser")
    LexerClass = getattr(lexer_mod, f"{grammar_name}Lexer")
    ParserClass = getattr(parser_mod, f"{grammar_name}Parser")
    return LexerClass, ParserClass


def parse_file(path: Path, LexerClass, ParserClass):
    from antlr4 import CommonTokenStream, FileStream
    from antlr4.error.ErrorListener import ErrorListener

    errors: list[tuple[int, int, str]] = []

    class Collector(ErrorListener):
        def syntaxError(self, recognizer, offendingSymbol, line, col, msg, e):
            errors.append((line, col, msg))

    listener = Collector()

    stream = FileStream(str(path), encoding="utf-8", errors="replace")
    lexer = LexerClass(stream)
    lexer.removeErrorListeners()
    lexer.addErrorListener(listener)

    tokens = CommonTokenStream(lexer)
    parser = ParserClass(tokens)
    parser.removeErrorListeners()
    parser.addErrorListener(listener)
    tree = parser.start()
    return errors, tree, parser


def main() -> int:
    seed = (
        Path(sys.argv[1])
        if len(sys.argv) > 1
        else REPO / "seeds" / "cuda-tile__Bytecode__attrsTest.mlir"
    )
    out = (
        Path(sys.argv[2])
        if len(sys.argv) > 2
        else REPO / "build" / "parse_tree_sample.txt"
    )

    ensure_parser()
    LexerClass, ParserClass = load_parser()
    errs, tree, parser = parse_file(seed, LexerClass, ParserClass)
    if errs:
        print(f"FAIL: parser errors in {seed}:", file=sys.stderr)
        for line, col, msg in errs[:5]:
            print(f"  line {line}:{col} {msg}", file=sys.stderr)
        return 1

    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(tree.toStringTree(recog=parser))
    print(f"wrote {out} ({out.stat().st_size:,} bytes) from {seed.name}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
