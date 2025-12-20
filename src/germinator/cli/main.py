"""
Main CLI entry point.
"""

import click
import logging

__version__ = "0.1.0"


@click.group()
@click.version_option(version=__version__)
def main():
    """Germinator: An extensible grammar-based fuzzer for IR-family languages."""
    pass


@main.command()
@click.option("--family", "-f", required=True, help="Mutation family (e.g., ssa.mlir)")
@click.option("--seeds", "-s", required=True, type=click.Path(exists=True), help="Seeds/trees directory")
@click.option("--output", "-o", default="./output", type=click.Path(), help="Output directory")
@click.option("--count", "-n", default=100, type=int, help="Number of tests to generate")
@click.option("--grammar", "-g", type=click.Path(exists=True), help="Path to processed grammar dir")
@click.option("--driver-config", type=click.Path(exists=True), help="Driver config file")
@click.option("--seed", type=int, help="Random seed for reproducibility")
@click.option("--max-depth", default=100, type=int, help="Maximum tree depth")
@click.option("--keep-trees/--no-keep-trees", default=True, help="Keep generated trees in population")
@click.option("--extension", "-e", default=".mlir", help="Output file extension")
@click.option("--no-generate", is_flag=True, help="Disable generate strategy")
@click.option("--no-mutate", is_flag=True, help="Disable mutate strategy")
@click.option("--no-recombine", is_flag=True, help="Disable recombine strategy")
@click.option("--no-edit", is_flag=True, help="Disable edit strategy")
@click.option("--no-insert", is_flag=True, help="Disable insert strategy")
@click.option("--k-ancestors", default=4, type=int, help="Context matching: k ancestors")
@click.option("--l-siblings", default=4, type=int, help="Context matching: l left siblings")
@click.option("--r-siblings", default=4, type=int, help="Context matching: r right siblings")
@click.option("--jobs", "-j", default=1, type=int, help="Number of parallel workers (0 = auto)")
@click.option("-v", "--verbose", is_flag=True, help="Verbose output")
def fuzz(family, seeds, output, count, grammar, driver_config, seed, max_depth,
         keep_trees, extension, no_generate, no_mutate, no_recombine, no_edit,
         no_insert, k_ancestors, l_siblings, r_siblings, jobs, verbose):
    """Generate test cases using grammar-based fuzzing."""
    import pickle
    import time
    from pathlib import Path
    from multiprocessing import Pool, cpu_count

    from germinator.families import get_family
    from germinator.seeds.population import Population
    from germinator.engine import Generator, GeneratorConfig
    from germinator.core import ContextMatcher

    # Setup logging
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
    )
    logger = logging.getLogger(__name__)

    # Auto-detect jobs
    if jobs <= 0:
        jobs = cpu_count()

    logger.info(f"Using {jobs} parallel workers")

    # Load family info
    logger.info(f"Loading mutation family: {family}")
    mutation_family = get_family(family)

    # Context matcher
    context_matcher = ContextMatcher(
        k_ancestors=k_ancestors,
        l_siblings=l_siblings,
        r_siblings=r_siblings,
    )

    # Load insert patterns
    insert_patterns = {}
    if grammar:
        grammar_path = Path(grammar)
        patterns_file = grammar_path / "insert_patterns.pkl"
        if patterns_file.exists():
            with open(patterns_file, "rb") as f:
                insert_patterns = pickle.load(f)
            logger.info(f"Loaded {len(insert_patterns)} insert patterns")

    # Setup output dir
    output_path = Path(output)
    output_path.mkdir(parents=True, exist_ok=True)

    seeds_path = Path(seeds)

    # Strategy flags
    enable_generate = not no_generate
    enable_mutate = not no_mutate
    enable_recombine = not no_recombine
    enable_edit = not no_edit
    enable_insert = not no_insert and bool(insert_patterns)

    logger.info(f"Strategies enabled: generate={enable_generate}, mutate={enable_mutate}, "
                f"recombine={enable_recombine}, edit={enable_edit}, insert={enable_insert}")

    start_time = time.time()

    if jobs == 1:
        # Single-threaded mode
        population = Population.from_directory(
            directory=seeds_path,
            context_matcher=context_matcher,
            valid_targets=mutation_family.config.valid_mutation_targets or None,
        )

        # Load generator factory
        generator_factory = None
        if grammar:
            import importlib.util
            generator_files = list(Path(grammar).glob("*Generator.py"))
            if generator_files:
                gen_path = generator_files[0]
                logger.info(f"Loading generator from: {gen_path}")
                spec = importlib.util.spec_from_file_location("generator", gen_path)
                module = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(module)

                for name in dir(module):
                    if name.endswith("Generator") and not name.startswith("_"):
                        generator_factory = getattr(module, name)
                        logger.info(f"Using generator class: {name}")
                        break

        config = GeneratorConfig(
            output_dir=output_path,
            output_pattern=f"test_%06d{extension}",
            max_depth=max_depth,
            keep_trees=keep_trees,
            enable_generate=enable_generate and generator_factory is not None,
            enable_mutate=enable_mutate and generator_factory is not None and population.can_mutate(),
            enable_recombine=enable_recombine and population.can_recombine(),
            enable_edit=enable_edit and population.can_recombine(),
            enable_insert=enable_insert and population.can_recombine(),
        )

        generator = Generator(
            family=mutation_family,
            population=population,
            generator_factory=generator_factory,
            insert_patterns=insert_patterns,
            config=config,
        )

        if seed is not None:
            generator.set_seed(seed)

        logger.info(f"Generating {count} test cases...")
        results = generator.generate(count)
        successful = sum(1 for _, r in results if r.success)

    else:
        # Parallel mode
        from germinator.engine.worker import generate_worker

        logger.info(f"Generating {count} test cases with {jobs} workers...")

        # Split work
        tests_per_worker = count // jobs
        remainder = count % jobs

        # Build args for each worker
        worker_args = []
        for i in range(jobs):
            n = tests_per_worker + (1 if i < remainder else 0)
            if n > 0:
                args = (
                    i,                      # worker_id
                    n,                      # num_tests
                    seed,                   # worker_seed
                    str(seeds_path),        # seeds_path
                    str(output_path),       # output_path
                    extension,              # extension
                    max_depth,              # max_depth
                    family,                 # family_name
                    k_ancestors,            # k_ancestors
                    l_siblings,             # l_siblings
                    r_siblings,             # r_siblings
                    str(grammar) if grammar else None,  # grammar_path
                    enable_generate,        # enable_generate
                    enable_mutate,          # enable_mutate
                    enable_recombine,       # enable_recombine
                    enable_edit,            # enable_edit
                    enable_insert,          # enable_insert
                    insert_patterns,        # insert_patterns
                )
                worker_args.append(args)

        # Run in parallel
        successful = 0
        with Pool(processes=jobs) as pool:
            results = pool.map(generate_worker, worker_args)
            successful = sum(results)

    elapsed = time.time() - start_time
    rate = successful / elapsed if elapsed > 0 else 0
    logger.info(f"Generated {successful} successful tests in {elapsed:.2f}s ({rate:.1f} tests/sec)")


if __name__ == "__main__":
    main()