"""CLI entry point for the driving scene description pipeline.

This module defines the argument parser and dispatches to the
appropriate pipeline functions based on the user's chosen command.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from src.pipeline import (
    get_run_dir,
    run_compare,
    run_evaluate,
    run_generate,
    run_temporal,
    setup_logging,
    _export_training_data,
)


def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Driving Scene Description Generator Pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Prepare dataset (after downloading BDD100K)
  python -m src.pipeline prepare --n 200

  # Generate descriptions with a specific prompt
  python -m src.pipeline generate --prompt v1_baseline --limit 10

  # Evaluate existing results
  python -m src.pipeline evaluate --results outputs/run_id/results_v1_baseline.json

  # Run full pipeline (generate + evaluate)
  python -m src.pipeline full --prompt v4_cot --limit 20

  # Compare all prompt variants
  python -m src.pipeline compare --limit 10

  # List available prompts
  python -m src.pipeline list-prompts
        """,
    )

    subparsers = parser.add_subparsers(dest="command", help="Pipeline command to run")

    # Prepare dataset
    prep = subparsers.add_parser("prepare", help="Prepare sampled dataset from BDD100K")
    prep.add_argument("--n", type=int, default=None, help="Number of images to sample")
    prep.add_argument("--seed", type=int, default=None, help="Random seed")
    prep.add_argument("--force", action="store_true", help="Force re-sampling")

    # Generate
    gen = subparsers.add_parser("generate", help="Generate scene descriptions")
    gen.add_argument("--prompt", required=True, help="Prompt variant ID")
    gen.add_argument("--limit", type=int, default=None, help="Max images to process")
    gen.add_argument("--run-id", default=None, help="Run ID for output directory")
    gen.add_argument("--dry-run", action="store_true", help="Print prompt without API calls")

    # Evaluate
    ev = subparsers.add_parser("evaluate", help="Evaluate generated results")
    ev.add_argument("--results", required=True, type=Path, help="Path to results JSON")
    ev.add_argument("--judge", action="store_true", help="Include LLM-as-judge")

    # Full pipeline
    full = subparsers.add_parser("full", help="Generate + evaluate end-to-end")
    full.add_argument("--prompt", required=True, help="Prompt variant ID")
    full.add_argument("--limit", type=int, default=None, help="Max images")
    full.add_argument("--judge", action="store_true", help="Include LLM-as-judge")

    # Analyze (AI Agent)
    analyze = subparsers.add_parser("analyze", help="AI agent: analyze errors and suggest improvements")
    analyze.add_argument("--results", required=True, type=Path, help="Path to evaluation JSON")
    analyze.add_argument("--prompt", default=None, help="Original prompt ID (for auto-improvement)")
    analyze.add_argument("--save", type=Path, default=None, help="Save report to JSON")

    # Compare
    comp = subparsers.add_parser("compare", help="Compare all prompt variants")
    comp.add_argument("--limit", type=int, default=None, help="Images per variant")
    comp.add_argument("--prompts", nargs="*", default=None, help="Specific prompt IDs")
    comp.add_argument("--judge", action="store_true", help="Include LLM-as-judge")

    # List prompts
    subparsers.add_parser("list-prompts", help="List available prompt variants")

    # Temporal analysis
    temp = subparsers.add_parser("temporal", help="Process video frame sequences for scene evolution")
    temp.add_argument("--prompt", default="v8_combined", help="Base prompt variant ID")
    temp.add_argument("--limit", type=int, default=None, help="Max sequences to process")

    # Export training data
    export = subparsers.add_parser("export-training", help="Export results as SFT training data")
    export.add_argument("--results", required=True, type=Path, help="Path to results JSON")
    export.add_argument("--output", type=Path, default=None, help="Output JSONL file path")
    export.add_argument("--prompt", default=None, help="Prompt variant ID (for system prompt)")

    parser.add_argument("-v", "--verbose", action="store_true", help="Verbose logging")

    args = parser.parse_args()
    setup_logging(args.verbose if hasattr(args, "verbose") else False)

    if args.command == "prepare":
        from src.data.downloader import prepare_dataset
        prepare_dataset(n=args.n, seed=args.seed, force=args.force)

    elif args.command == "generate":
        run_generate(
            prompt_id=args.prompt,
            limit=args.limit,
            run_dir=get_run_dir(args.run_id) if args.run_id else None,
            dry_run=args.dry_run,
        )

    elif args.command == "evaluate":
        run_evaluate(results_file=args.results, use_judge=args.judge)

    elif args.command == "full":
        run_dir = run_generate(prompt_id=args.prompt, limit=args.limit)
        results_file = run_dir / f"results_{args.prompt}.json"
        run_evaluate(results_file=results_file, run_dir=run_dir, use_judge=args.judge)

    elif args.command == "compare":
        run_compare(limit=args.limit, prompt_ids=args.prompts, use_judge=args.judge)

    elif args.command == "analyze":
        from src.agent.analyzer import ErrorAnalyzerAgent
        agent = ErrorAnalyzerAgent()
        report = agent.analyze(args.results)
        agent.print_report(report)
        if args.save:
            agent.save_report(report, args.save)
        if args.prompt:
            from src.prompts.templates import get_prompt
            original = get_prompt(args.prompt)
            improved = agent.generate_improved_prompt(original, report)
            print("\n[AUTO-IMPROVED] Prompt:")
            print("-" * 60)
            print(improved)
            print("-" * 60)

    elif args.command == "list-prompts":
        from src.prompts.templates import list_prompts
        prompts = list_prompts()
        print(f"\n{'ID':<20} {'Strategy':<30} {'Description'}")
        print("-" * 80)
        for p in prompts:
            print(f"{p['id']:<20} {p['strategy']:<30} {p['description']}")
        print()

    elif args.command == "export-training":
        _export_training_data(args.results, args.output, args.prompt)

    elif args.command == "temporal":
        run_temporal(prompt_id=args.prompt, limit=args.limit)

    else:
        parser.print_help()


if __name__ == "__main__":
    main()
