"""
Parallel worker for test generation.
"""

from pathlib import Path


def generate_worker(args):
    """
    Worker function for parallel generation.
    
    Args is a tuple of all necessary parameters since multiprocessing
    requires a single argument.
    """
    (
        worker_id,
        num_tests,
        worker_seed,
        seeds_path,
        output_path,
        extension,
        max_depth,
        family_name,
        k_ancestors,
        l_siblings,
        r_siblings,
        grammar_path,
        enable_generate,
        enable_mutate,
        enable_recombine,
        enable_edit,
        enable_insert,
        insert_patterns,
    ) = args

    # Import here to avoid issues with multiprocessing
    from germinator.families import get_family
    from germinator.seeds.population import Population
    from germinator.engine import Generator, GeneratorConfig
    from germinator.core import ContextMatcher

    # Load family
    family = get_family(family_name)

    # Create context matcher
    context_matcher = ContextMatcher(
        k_ancestors=k_ancestors,
        l_siblings=l_siblings,
        r_siblings=r_siblings,
    )

    # Load population
    population = Population.from_directory(
        directory=Path(seeds_path),
        context_matcher=context_matcher,
        valid_targets=family.config.valid_mutation_targets or None,
    )

    # Load generator factory if grammar provided
    generator_factory = None
    if grammar_path:
        import importlib.util
        grammar_p = Path(grammar_path)
        generator_files = list(grammar_p.glob("*Generator.py"))
        if generator_files:
            gen_path = generator_files[0]
            spec = importlib.util.spec_from_file_location("generator", gen_path)
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)

            for name in dir(module):
                if name.endswith("Generator") and not name.startswith("_"):
                    generator_factory = getattr(module, name)
                    break

    # Create config
    config = GeneratorConfig(
        output_dir=Path(output_path),
        output_pattern=f"test_{worker_id:02d}_%06d{extension}",
        max_depth=max_depth,
        keep_trees=False,
        enable_generate=enable_generate and generator_factory is not None,
        enable_mutate=enable_mutate and generator_factory is not None and population.can_mutate(),
        enable_recombine=enable_recombine and population.can_recombine(),
        enable_edit=enable_edit and population.can_recombine(),
        enable_insert=enable_insert and population.can_recombine(),
    )

    # Create generator
    generator = Generator(
        family=family,
        population=population,
        generator_factory=generator_factory,
        insert_patterns=insert_patterns or {},
        config=config,
    )

    if worker_seed is not None:
        generator.set_seed(worker_seed + worker_id)

    # Generate
    results = generator.generate(num_tests)
    successful = sum(1 for _, r in results if r.success)

    return successful