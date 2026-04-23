#!/usr/bin/env python3
"""
generate_insert_patterns.py — Extract insert patterns from an ANTLR .g4 grammar
and emit a C++ header the fuzzer can compile against.

Supports:
  - Simple quantified rule refs:    `foo+`, `bar?`              → Quantifier
  - Comma-separated lists:          `foo (',' foo)*`            → Quantifier
    (or any multi-symbol repeated unit)
  - Alternation under a quantifier: `(foo | bar)+`              → Quantifier with
                                                                  multiple AltBranches
  - Unquantified alternation:       `(foo | bar)` (no suffix)   → bare AltBranch
  - Literals and single rule refs:                              → bare Symbol

A rule is extracted iff we can model its entire body with the shapes above.
Anything else (nested groups we can't flatten, free-form structure) causes the
rule to be skipped.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from math import inf
from pathlib import Path

import click

# ---------------------------------------------------------------------------
# Data model — mirrors the (to-be-updated) C++ representation.
# ---------------------------------------------------------------------------


@dataclass
class Symbol:
    is_rule: bool          # True if rule_ref, False if string literal
    value: str             # rule name or literal text


@dataclass
class AltBranch:
    """One alternative in an alternation — a flat sequence of symbols."""
    symbols: list[Symbol] = field(default_factory=list)


@dataclass
class QuantifierSpec:
    min: int
    max: float             # inf for unbounded
    alternatives: list[AltBranch] = field(default_factory=list)


# MatchElement = Symbol | AltBranch | QuantifierSpec


@dataclass
class InsertPattern:
    match_pattern: list = field(default_factory=list)
    child_rules: set = field(default_factory=set)


# ---------------------------------------------------------------------------
# Grammar parsing — intermediate RuleChild representation.
# ---------------------------------------------------------------------------


@dataclass
class RuleChild:
    kind: str              # literal | rule_ref | token_ref | quantifier |
                           # alternation | opaque
    value: str = ""
    quantifier_min: int = 1
    quantifier_max: float = 1
    quantifier_body: "list[RuleChild] | None" = None
    quantifier_alts: "list[list[RuleChild]] | None" = None
    alt_branches: "list[list[RuleChild]] | None" = None


QUANT_MAP = {"?": (0, 1), "*": (0, inf), "+": (1, inf)}

RULE_RE = re.compile(r"^([a-z_][a-zA-Z_0-9]*)\s*:\s*(.*?)\s*;", re.MULTILINE | re.DOTALL)


def _strip_comments(src: str) -> str:
    src = re.sub(r"//[^\n]*", "", src)
    src = re.sub(r"/\*.*?\*/", "", src, flags=re.DOTALL)
    return src


def _split_top_level_pipe(body: str) -> list[str]:
    parts: list[str] = []
    depth = 0
    start = 0
    for i, ch in enumerate(body):
        if ch in "([{<":
            depth += 1
        elif ch in ")]}>":
            depth -= 1
        elif ch == "|" and depth == 0:
            parts.append(body[start:i])
            start = i + 1
    parts.append(body[start:])
    return parts


def _consume_quantifier(body: str, i: int) -> str | None:
    if i < len(body) and body[i] in "?*+":
        return body[i]
    return None


def parse_rule_body(body: str) -> list[RuleChild] | None:
    children: list[RuleChild] = []
    i = 0
    body = body.strip()

    while i < len(body):
        ch = body[i]

        if ch.isspace():
            i += 1
            continue

        # String literal.
        if ch == "'":
            end = body.index("'", i + 1)
            literal = body[i + 1:end]
            i = end + 1
            quant = _consume_quantifier(body, i)
            if quant is not None:
                qmin, qmax = QUANT_MAP[quant]
                children.append(RuleChild(
                    kind="quantifier",
                    quantifier_min=qmin, quantifier_max=qmax,
                    quantifier_body=[RuleChild(kind="literal", value=literal)],
                ))
                i += 1
            else:
                children.append(RuleChild(kind="literal", value=literal))
            continue

        # Parenthesized group.
        if ch == "(":
            start = i + 1
            depth = 1
            i += 1
            while i < len(body) and depth > 0:
                if body[i] == "(":
                    depth += 1
                elif body[i] == ")":
                    depth -= 1
                i += 1
            group_body = body[start:i - 1].strip()
            quant = _consume_quantifier(body, i)
            if quant is not None:
                i += 1

            pipe_parts = _split_top_level_pipe(group_body)
            is_alternation = len(pipe_parts) > 1

            branches: list[list[RuleChild]] = []
            any_opaque = False
            for alt in pipe_parts:
                sub = parse_rule_body(alt)
                if sub is None or any(c.kind == "opaque" for c in sub):
                    any_opaque = True
                    break
                branches.append(sub)

            if any_opaque:
                children.append(RuleChild(kind="opaque", value=group_body))
                continue

            if quant is not None:
                qmin, qmax = QUANT_MAP[quant]
                children.append(RuleChild(
                    kind="quantifier",
                    quantifier_min=qmin, quantifier_max=qmax,
                    quantifier_alts=branches if is_alternation else None,
                    quantifier_body=branches[0] if not is_alternation else None,
                ))
                continue

            if is_alternation:
                children.append(RuleChild(kind="alternation", alt_branches=branches))
            else:
                # Unquantified non-alternation group: parens are semantically inert.
                children.extend(branches[0])
            continue

        # Identifier (rule_ref or token_ref).
        m = re.match(r"[A-Za-z_][A-Za-z_0-9]*", body[i:])
        if m:
            name = m.group(0)
            i += len(name)
            quant = _consume_quantifier(body, i)
            if quant is not None:
                i += 1
            is_parser_rule = name[0].islower() or name[0] == "_"
            inner_kind = "rule_ref" if is_parser_rule else "token_ref"
            inner = RuleChild(kind=inner_kind, value=name)
            if quant is not None:
                qmin, qmax = QUANT_MAP[quant]
                children.append(RuleChild(
                    kind="quantifier",
                    quantifier_min=qmin, quantifier_max=qmax,
                    quantifier_body=[inner],
                ))
            else:
                children.append(inner)
            continue

        i += 1

    return children


def _has_top_level_alt(body: str) -> bool:
    return len(_split_top_level_pipe(body)) > 1


def parse_grammar(path: Path) -> tuple[dict[str, list[RuleChild]], list[str]]:
    src = _strip_comments(path.read_text())
    rules: dict[str, list[RuleChild]] = {}
    skipped: list[str] = []

    for m in RULE_RE.finditer(src):
        name = m.group(1)
        body = m.group(2).strip()

        if _has_top_level_alt(body):
            # Represent as a single alternation RuleChild; treated like a
            # bare alternation in the emitted pattern.
            alts = _split_top_level_pipe(body)
            branches: list[list[RuleChild]] = []
            ok = True
            for alt in alts:
                sub = parse_rule_body(alt)
                if sub is None or any(c.kind == "opaque" for c in sub):
                    ok = False
                    break
                branches.append(sub)
            if not ok:
                skipped.append(f"{name} (opaque alternation)")
                continue
            rules[name] = [RuleChild(kind="alternation", alt_branches=branches)]
            continue

        children = parse_rule_body(body)
        if children is None:
            skipped.append(f"{name} (parse failed)")
            continue
        if any(c.kind == "opaque" for c in children):
            skipped.append(f"{name} (opaque)")
            continue
        rules[name] = children

    return rules, skipped


# ---------------------------------------------------------------------------
# RuleChild → MatchElement conversion.
# ---------------------------------------------------------------------------


def _rc_to_symbols(rc_list: list[RuleChild]) -> list[Symbol] | None:
    symbols: list[Symbol] = []
    for c in rc_list:
        if c.kind == "literal":
            symbols.append(Symbol(is_rule=False, value=c.value))
        elif c.kind in ("rule_ref", "token_ref"):
            symbols.append(Symbol(is_rule=True, value=c.value))
        else:
            return None
    return symbols


def _rc_to_alt_branches(branches_rc: list[list[RuleChild]]) -> list[AltBranch] | None:
    out: list[AltBranch] = []
    for br in branches_rc:
        syms = _rc_to_symbols(br)
        if syms is None:
            return None
        out.append(AltBranch(symbols=syms))
    return out


def _rc_to_match_elements(children: list[RuleChild]) -> list | None:
    out: list = []
    for c in children:
        if c.kind == "literal":
            out.append(Symbol(is_rule=False, value=c.value))
        elif c.kind in ("rule_ref", "token_ref"):
            out.append(Symbol(is_rule=True, value=c.value))
        elif c.kind == "alternation":
            branches = _rc_to_alt_branches(c.alt_branches or [])
            if branches is None:
                return None
            # Unquantified alternation: emit as QuantifierSpec{1,1,branches}.
            # The C++ side treats min=max=1 with multiple alternatives as
            # "match exactly one of these branches at this position."
            out.append(QuantifierSpec(min=1, max=1, alternatives=branches))
        elif c.kind == "quantifier":
            if c.quantifier_alts is not None:
                branches = _rc_to_alt_branches(c.quantifier_alts)
            elif c.quantifier_body is not None:
                syms = _rc_to_symbols(c.quantifier_body)
                if syms is None:
                    return None
                branches = [AltBranch(symbols=syms)]
            else:
                return None
            if branches is None:
                return None
            out.append(QuantifierSpec(
                min=c.quantifier_min, max=c.quantifier_max, alternatives=branches,
            ))
        else:
            return None
    return out


# ---------------------------------------------------------------------------
# Rule selection: which rules become insert patterns.
# ---------------------------------------------------------------------------


def extract_insert_patterns(
    rules: dict[str, list[RuleChild]],
) -> tuple[dict[str, InsertPattern], list[str]]:
    patterns: dict[str, InsertPattern] = {}
    skipped: list[str] = []

    for name, children in rules.items():
        elements = _rc_to_match_elements(children)
        if elements is None:
            skipped.append(f"{name} (unmodelable)")
            continue

        # Insert slot = a quantifier with min != max. Degenerate {1,1}
        # alternations don't add new structure; they're just pick-one.
        has_slot = any(
            isinstance(e, QuantifierSpec) and (e.min, e.max) != (1, 1)
            for e in elements
        )
        if not has_slot:
            skipped.append(f"{name} (no insertion slot)")
            continue

        pattern = InsertPattern()
        pattern.match_pattern = elements
        for e in elements:
            _collect_child_rules(e, pattern.child_rules)
        patterns[name] = pattern

    return patterns, skipped


def _collect_child_rules(elem, into: set) -> None:
    if isinstance(elem, Symbol):
        if elem.is_rule:
            into.add(elem.value)
    elif isinstance(elem, AltBranch):
        for s in elem.symbols:
            _collect_child_rules(s, into)
    elif isinstance(elem, QuantifierSpec):
        for br in elem.alternatives:
            _collect_child_rules(br, into)


# ---------------------------------------------------------------------------
# Codegen.
# ---------------------------------------------------------------------------


def _escape(s: str) -> str:
    return s.replace("\\", "\\\\").replace('"', '\\"')


def _emit_symbol(sym: Symbol) -> str:
    kind = "true" if sym.is_rule else "false"
    return f'Symbol{{{kind}, "{_escape(sym.value)}"}}'


def _emit_alt_branch(br: AltBranch) -> str:
    syms = ", ".join(_emit_symbol(s) for s in br.symbols)
    return f"AltBranch{{{{{syms}}}}}"


def _emit_quantifier(q: QuantifierSpec) -> str:
    maxv = "INT_MAX" if q.max == inf else str(int(q.max))
    alts = ", ".join(_emit_alt_branch(a) for a in q.alternatives)
    return f"QuantifierSpec{{{int(q.min)}, {maxv}, {{{alts}}}}}"


def _emit_match_element(elem) -> str:
    if isinstance(elem, Symbol):
        return f"MatchElement{{{_emit_symbol(elem)}}}"
    if isinstance(elem, AltBranch):
        return f"MatchElement{{{_emit_alt_branch(elem)}}}"
    if isinstance(elem, QuantifierSpec):
        return f"MatchElement{{{_emit_quantifier(elem)}}}"
    raise ValueError(f"Unknown match element: {elem!r}")


def emit_header(patterns: dict[str, InsertPattern], output_path: Path) -> None:
    lines = [
        "// AUTO-GENERATED — do not edit by hand.",
        "// Regenerate with scripts/build/generate_insert_patterns.py",
        f"// {len(patterns)} patterns extracted.",
        "//",
        "// Data model (to be mirrored in insert.h):",
        "//   Symbol{is_rule, value}",
        "//   AltBranch{symbols: vector<Symbol>}",
        "//   QuantifierSpec{min, max, alternatives: vector<AltBranch>}",
        "//   MatchElement = variant<Symbol, AltBranch, QuantifierSpec>",
        "//",
        "// min=max=1 with multiple alternatives means 'pick one branch',",
        "// used to represent unquantified alternations at a position.",
        "",
        "#pragma once",
        "",
        '#include "insert.h"',
        "",
        "#include <climits>",
        "#include <string>",
        "",
        "namespace mlir_fuzzer {",
        "",
        "inline InsertPatterns get_insert_patterns() {",
        "  InsertPatterns patterns;",
    ]
    for name in sorted(patterns):
        pat = patterns[name]
        lines.append("")
        lines.append(f"  // {name}")
        lines.append("  {")
        lines.append("    InsertPattern p;")
        for elem in pat.match_pattern:
            lines.append(f"    p.match_pattern.push_back({_emit_match_element(elem)});")
        for rule in sorted(pat.child_rules):
            lines.append(f'    p.child_rules.insert("{_escape(rule)}");')
        lines.append(f'    patterns["{_escape(name)}"] = std::move(p);')
        lines.append("  }")
    lines.extend([
        "",
        "  return patterns;",
        "}",
        "",
        "} // namespace mlir_fuzzer",
        "",
    ])
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(lines))
    click.echo(f"Wrote {len(patterns)} patterns to {output_path}")


# ---------------------------------------------------------------------------
# Pretty-print for --verbose.
# ---------------------------------------------------------------------------


def _pp_symbol(s: Symbol) -> str:
    return f"<{s.value}>" if s.is_rule else f"'{s.value}'"


def _pp_branch(b: AltBranch) -> str:
    return " ".join(_pp_symbol(s) for s in b.symbols)


def _pp_elem(e) -> str:
    if isinstance(e, Symbol):
        return _pp_symbol(e)
    if isinstance(e, AltBranch):
        return f"[ {_pp_branch(e)} ]"
    if isinstance(e, QuantifierSpec):
        mx = "∞" if e.max == inf else str(int(e.max))
        suffix = f"{{{e.min},{mx}}}"
        if len(e.alternatives) == 1:
            return f"({_pp_branch(e.alternatives[0])}){suffix}"
        alts = " | ".join(_pp_branch(a) for a in e.alternatives)
        return f"({alts}){suffix}"
    return repr(e)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


@click.command()
@click.option("--grammar", "-g", required=True,
              type=click.Path(exists=True, path_type=Path),
              help="Path to the ANTLR .g4 grammar")
@click.option("--output", "-o", required=True,
              type=click.Path(path_type=Path),
              help="Output insert_patterns.h")
@click.option("--verbose", "-v", is_flag=True)
def main(grammar: Path, output: Path, verbose: bool) -> None:
    click.echo(f"Parsing grammar: {grammar}")
    rules, parse_skipped = parse_grammar(grammar)
    click.echo(f"  {len(rules)} parser rules parsed")
    if parse_skipped:
        click.echo(f"  {len(parse_skipped)} rules skipped at parse stage:")
        for s in parse_skipped[:10]:
            click.echo(f"    - {s}")
        if len(parse_skipped) > 10:
            click.echo(f"    ... +{len(parse_skipped) - 10} more")

    patterns, extract_skipped = extract_insert_patterns(rules)
    click.echo(f"Extracted {len(patterns)} insert patterns")
    if extract_skipped:
        click.echo(f"  {len(extract_skipped)} rules skipped at extraction:")
        for s in extract_skipped[:10]:
            click.echo(f"    - {s}")
        if len(extract_skipped) > 10:
            click.echo(f"    ... +{len(extract_skipped) - 10} more")

    if verbose:
        for name in sorted(patterns):
            pat = patterns[name]
            click.echo(f"\n  {name}:")
            for elem in pat.match_pattern:
                click.echo(f"    {_pp_elem(elem)}")
            click.echo(f"    child_rules: {sorted(pat.child_rules)}")

    emit_header(patterns, output)


if __name__ == "__main__":
    main()