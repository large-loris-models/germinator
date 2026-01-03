"""
Grammar processing CLI.
"""

import click


@click.command()
@click.argument("grammar", nargs=-1, required=True, type=click.Path(exists=True))
@click.option("--output", "-o", default="./build", type=click.Path(), help="Output directory")
@click.option("--rule", "-r", help="Default rule for generation")
@click.option("--encoding", default="utf-8", help="Grammar file encoding")
@click.option("-v", "--verbose", is_flag=True, help="Verbose output")
def main(grammar, output, rule, encoding, verbose):
    """Process ANTLR4 grammar files for use with Germinator."""
    import logging
    from pathlib import Path
    
    from germinator.grammar import GrammarProcessor
    
    # Setup logging
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
    )
    logger = logging.getLogger(__name__)
    
    # Process
    processor = GrammarProcessor(output_dir=Path(output))
    
    grammar_files = [Path(g) for g in grammar]
    logger.info(f"Processing grammars: {[g.name for g in grammar_files]}")
    
    result = processor.process(
        grammar_files=grammar_files,
        default_rule=rule,
        encoding=encoding,
    )
    
    logger.info(f"Generated: {result.generator_path}")
    logger.info(f"Grammar name: {result.grammar_name}")
    logger.info(f"Default rule: {result.default_rule}")
    logger.info(f"Insert patterns: {len(result.insert_patterns)}")


if __name__ == "__main__":
    main()