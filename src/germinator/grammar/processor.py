"""
Grammar processing.

Parses ANTLR4 grammars and extracts:
- Generator code (via Grammarinator)
- Insert patterns for SynthFuzz-style mutations
- Grammar graph for analysis
"""

from __future__ import annotations

import logging
import pickle
import re
from collections import OrderedDict, defaultdict
from dataclasses import dataclass, field
from math import inf
from pathlib import Path
from typing import Any

from germinator.grammar.patterns import InsertPattern, QuantifierSpec

logger = logging.getLogger(__name__)


# ============================================================================
# Graph Node Classes (ported from SynthFuzz)
# ============================================================================

class Node:
    """Base node in the grammar graph."""
    _cnt = 0

    def __init__(self, id=None):
        if id is None:
            id = Node._cnt
            Node._cnt += 1
        self.id = id
        self.out_neighbours = []

    def __str__(self):
        return f"{self.__class__.__name__}(id={self.id})"


class RuleNode(Node):
    """Base class for grammar rules."""
    
    def __init__(self, name: str, rule_type: str):
        super().__init__(id=name)
        self.name = name
        self.type = rule_type
        self.min_depth = None


class UnparserRuleNode(RuleNode):
    """Parser rule node."""
    
    def __init__(self, name: str):
        super().__init__(name, "UnparserRule")


class UnlexerRuleNode(RuleNode):
    """Lexer rule node."""
    
    def __init__(self, name: str):
        super().__init__(name, "UnlexerRule")


class QuantifierNode(Node):
    """Quantified element (*, +, ?)."""
    
    def __init__(self, min_val: int, max_val: int | float):
        super().__init__()
        self.min = min_val
        self.max = max_val

    def __str__(self):
        return f"QuantifierNode(min={self.min}, max={self.max})"


class LiteralNode(Node):
    """Literal string node."""
    
    def __init__(self, src: str):
        super().__init__()
        self.src = src

    def __str__(self):
        return f"LiteralNode(src={self.src!r})"


class AlternationNode(Node):
    """Alternation (|) node."""
    
    def __init__(self):
        super().__init__()


# ============================================================================
# Grammar Graph
# ============================================================================

@dataclass
class GrammarGraph:
    """Internal representation of the grammar."""
    name: str = None
    default_rule: str = None
    rules: dict[str, RuleNode] = field(default_factory=dict)
    
    def get_parser_rules(self) -> list[UnparserRuleNode]:
        """Get all parser rules."""
        return [r for r in self.rules.values() if isinstance(r, UnparserRuleNode)]


# ============================================================================
# Processor Result
# ============================================================================

@dataclass
class ProcessorResult:
    """Result of grammar processing."""
    generator_path: Path
    graph_path: Path | None
    patterns_path: Path | None
    grammar_name: str
    default_rule: str | None
    insert_patterns: dict[str, InsertPattern]

    def load_generator_factory(self):
        """Load the generated generator class."""
        import importlib.util

        spec = importlib.util.spec_from_file_location(
            self.grammar_name, self.generator_path
        )
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)

        # Find the generator class
        for name in dir(module):
            if name.endswith("Generator") and not name.startswith("_"):
                return getattr(module, name)
        
        raise RuntimeError(f"No generator class found in {self.generator_path}")


# ============================================================================
# Grammar Processor
# ============================================================================

class GrammarProcessor:
    """
    Processes ANTLR4 grammars for use with Germinator.

    This is a port of SynthFuzz's ProcessorTool that generates
    insert patterns for parameterized mutations.
    """

    def __init__(self, output_dir: Path | str):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def process(
        self,
        grammar_files: list[Path | str],
        default_rule: str | None = None,
        encoding: str = "utf-8",
    ) -> ProcessorResult:
        """
        Process grammar files and generate outputs.
        
        Args:
            grammar_files: Paths to .g4 grammar files
            default_rule: Default rule for generation
            encoding: Grammar file encoding
        
        Returns:
            ProcessorResult with paths to generated files
        """
        from grammarinator.tool.processor import ProcessorTool as GrammarinatorProcessor

        # Step 1: Use Grammarinator to generate the Python generator
        logger.info("Running Grammarinator processor...")
        processor = GrammarinatorProcessor(lang="py", work_dir=str(self.output_dir))

        grammar_paths = [str(p) for p in grammar_files]

        processor.process(
            grammar_paths,
            default_rule=default_rule,
            encoding=encoding,
            actions=True,
            pep8=True,
        )

        # Step 2: Build our own grammar graph for pattern extraction
        logger.info("Building grammar graph...")
        graph = self._build_graph(grammar_paths, default_rule, encoding)

        # Save graph
        graph_path = self.output_dir / "graph.pkl"
        with open(graph_path, "wb") as f:
            pickle.dump(graph, f)
        logger.info(f"Saved graph to {graph_path}")

        # Step 3: Extract insert patterns
        logger.info("Extracting insert patterns...")
        insert_patterns = self._derive_insert_patterns(graph)

        patterns_path = self.output_dir / "insert_patterns.pkl"
        with open(patterns_path, "wb") as f:
            pickle.dump(insert_patterns, f)
        logger.info(f"Saved {len(insert_patterns)} insert patterns to {patterns_path}")

        # Find generator file
        generator_files = list(self.output_dir.glob("*Generator.py"))
        if not generator_files:
            generator_files = list(self.output_dir.glob("*.py"))
            generator_files = [f for f in generator_files if not f.name.startswith("_")]

        if not generator_files:
            raise RuntimeError(f"No generator file found in {self.output_dir}")

        generator_path = generator_files[0]
        grammar_name = generator_path.stem

        logger.info(f"Generator: {generator_path}")
        logger.info(f"Default rule: {graph.default_rule}")
        logger.info(f"Insert patterns: {len(insert_patterns)}")

        return ProcessorResult(
            generator_path=generator_path,
            graph_path=graph_path,
            patterns_path=patterns_path,
            grammar_name=grammar_name,
            default_rule=graph.default_rule,
            insert_patterns=insert_patterns,
        )

    def _build_graph(
        self,
        grammar_paths: list[str],
        default_rule: str | None,
        encoding: str,
    ) -> GrammarGraph:
        """Build grammar graph by parsing the ANTLR grammar."""
        from antlr4 import CommonTokenStream, FileStream
        from grammarinator.tool.g4 import ANTLRv4Lexer, ANTLRv4Parser

        graph = GrammarGraph()
        lexer_root, parser_root = None, None

        # Parse all grammar files
        for grammar_path in grammar_paths:
            antlr_parser = ANTLRv4Parser(
                CommonTokenStream(
                    ANTLRv4Lexer(FileStream(grammar_path, encoding=encoding))
                )
            )
            root = antlr_parser.grammarSpec()

            # Determine if lexer or parser grammar
            grammar_type = root.grammarDecl().grammarType()
            if grammar_type.LEXER() or not grammar_type.PARSER():
                lexer_root = root
            else:
                parser_root = root

        # Process both roots
        for root in [lexer_root, parser_root]:
            if root:
                self._process_grammar_root(root, graph, default_rule)

        return graph

    def _process_grammar_root(
        self,
        root,
        graph: GrammarGraph,
        default_rule: str | None,
    ):
        """Process a grammar root node."""
        # Extract grammar name
        if not graph.name:
            ident = root.grammarDecl().identifier()
            name = str(ident.TOKEN_REF() or ident.RULE_REF())
            graph.name = re.sub(r"^(.+?)(Lexer|Parser)?$", r"\1Generator", name)

        first_parser_rule = None

        # Process rules
        for rule in root.rules().ruleSpec():
            if rule.parserRuleSpec():
                rule_spec = rule.parserRuleSpec()
                rule_name = str(rule_spec.RULE_REF())
                rule_node = UnparserRuleNode(name=rule_name)

                if first_parser_rule is None:
                    first_parser_rule = rule_name

                # Extract rule structure
                self._extract_rule_structure(rule_spec.ruleBlock(), rule_node)
                graph.rules[rule_name] = rule_node

            elif rule.lexerRuleSpec():
                rule_spec = rule.lexerRuleSpec()
                rule_name = str(rule_spec.TOKEN_REF())
                rule_node = UnlexerRuleNode(name=rule_name)
                graph.rules[rule_name] = rule_node

        # Set default rule
        if default_rule:
            graph.default_rule = default_rule
        elif first_parser_rule:
            graph.default_rule = first_parser_rule

    def _extract_rule_structure(self, node, rule_node: UnparserRuleNode):
        """Extract the structure of a rule for pattern matching."""
        from grammarinator.tool.g4 import ANTLRv4Parser
        from antlr4 import ParserRuleContext

        if node is None:
            return

        # Handle different node types
        if isinstance(node, (
            ANTLRv4Parser.RuleBlockContext,
            ANTLRv4Parser.RuleAltListContext,
        )):
            children = [c for c in node.children if isinstance(c, ParserRuleContext)]
            
            # Check for alternation
            if isinstance(node, ANTLRv4Parser.RuleAltListContext) and len(children) > 1:
                alt_node = AlternationNode()
                rule_node.out_neighbours.append(alt_node)
                return
            
            for child in children:
                self._extract_rule_structure(child, rule_node)

        elif isinstance(node, ANTLRv4Parser.LabeledAltContext):
            self._extract_rule_structure(node.alternative(), rule_node)

        elif isinstance(node, ANTLRv4Parser.AlternativeContext):
            for element in node.element():
                self._extract_rule_structure(element, rule_node)

        elif isinstance(node, ANTLRv4Parser.ElementContext):
            # Check for quantifier suffix
            suffix = None
            if node.ebnfSuffix():
                suffix = str(node.ebnfSuffix().children[0])
            elif hasattr(node, "ebnf") and node.ebnf() and node.ebnf().blockSuffix():
                suffix = str(node.ebnf().blockSuffix().ebnfSuffix().children[0])

            if suffix:
                # Quantified element
                quant_ranges = {
                    "?": (0, 1),
                    "*": (0, inf),
                    "+": (1, inf),
                }
                min_val, max_val = quant_ranges.get(suffix, (1, 1))
                quant_node = QuantifierNode(min_val, max_val)

                # Get the inner element
                inner = node.children[0]
                inner_node = self._get_element_as_node(inner)
                
                if inner_node:
                    quant_node.out_neighbours.append(inner_node)
                    rule_node.out_neighbours.append(quant_node)
            else:
                # Non-quantified element
                elem_node = self._get_element_as_node(node.children[0])
                if elem_node:
                    rule_node.out_neighbours.append(elem_node)

        elif isinstance(node, ParserRuleContext) and node.getChildCount():
            for child in node.children:
                if isinstance(child, ParserRuleContext):
                    self._extract_rule_structure(child, rule_node)

    def _get_element_as_node(self, node) -> Node | None:
        """Convert an ANTLR element to a graph node."""
        from grammarinator.tool.g4 import ANTLRv4Parser

        if isinstance(node, ANTLRv4Parser.RulerefContext):
            return UnparserRuleNode(name=str(node.RULE_REF()))

        elif isinstance(node, ANTLRv4Parser.AtomContext):
            if node.ruleref():
                return UnparserRuleNode(name=str(node.ruleref().RULE_REF()))
            elif node.terminal():
                return self._get_element_as_node(node.terminal())
            # DOT, notSet, etc. - skip for now
            return None

        elif isinstance(node, ANTLRv4Parser.TerminalContext):
            if node.TOKEN_REF():
                return UnlexerRuleNode(name=str(node.TOKEN_REF()))
            elif node.STRING_LITERAL():
                src = str(node.STRING_LITERAL())[1:-1]  # Remove quotes
                return LiteralNode(src=src)

        elif isinstance(node, ANTLRv4Parser.BlockContext):
            # Blocks are complex - skip for simple pattern extraction
            return None

        elif isinstance(node, ANTLRv4Parser.EbnfContext):
            # Handle ebnf blocks
            if node.block():
                return self._get_element_as_node(node.block())
            return None

        return None

    def _derive_insert_patterns(self, graph: GrammarGraph) -> dict[str, InsertPattern]:
        """
        Derive insert patterns from grammar graph.
        
        This finds rules with quantifiers and creates patterns that
        describe where insertions can happen.
        """
        insert_patterns = {}
        parser_rules = graph.get_parser_rules()
        
        logger.debug(f"Analyzing {len(parser_rules)} parser rules for insert patterns")

        def is_simple_quantifier(node: QuantifierNode) -> bool:
            """Check if quantifier contains a single parser rule."""
            return (
                len(node.out_neighbours) == 1 and
                isinstance(node.out_neighbours[0], UnparserRuleNode)
            )

        def contains_simple_quantifier(rule: UnparserRuleNode) -> bool:
            """Check if rule has any simple quantifiers."""
            quantifiers = [
                n for n in rule.out_neighbours
                if isinstance(n, QuantifierNode)
            ]
            if not quantifiers:
                return False
            return any(is_simple_quantifier(q) for q in quantifiers)

        for rule in parser_rules:
            # Skip rules without quantifiers
            if not contains_simple_quantifier(rule):
                continue

            # Skip rules with alternations (too complex for now)
            has_alternation = any(
                isinstance(n, AlternationNode) for n in rule.out_neighbours
            )
            if has_alternation:
                logger.debug(f"Skipping {rule.name}: has alternation")
                continue

            # Build match pattern
            match_pattern = []
            child_rules = set()
            valid = True

            for child in rule.out_neighbours:
                if isinstance(child, QuantifierNode):
                    if not is_simple_quantifier(child):
                        valid = False
                        break

                    inner = child.out_neighbours[0]
                    match_pattern.append(QuantifierSpec(
                        min=child.min,
                        max=child.max,
                        rule_name=inner.name,
                    ))
                    child_rules.add(inner.name)

                elif isinstance(child, UnparserRuleNode):
                    match_pattern.append(child.name)

                elif isinstance(child, UnlexerRuleNode):
                    match_pattern.append(child.name)

                elif isinstance(child, LiteralNode):
                    match_pattern.append(child.src)

                elif isinstance(child, AlternationNode):
                    valid = False
                    break

                else:
                    logger.debug(f"Skipping {rule.name}: unexpected child type {type(child)}")
                    valid = False
                    break

            if valid and child_rules:
                insert_patterns[rule.name] = InsertPattern(match_pattern, child_rules)
                logger.debug(f"Added insert pattern for {rule.name}: {child_rules}")

        return insert_patterns