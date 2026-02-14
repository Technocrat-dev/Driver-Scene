"""End-to-end pipeline orchestrator for driving scene description generation.

Modes:
  generate  - Process images with a VLM prompt → JSON results
  evaluate  - Score existing results against ground truth
  full      - Generate + evaluate end-to-end
  compare   - Run all prompt variants on a subset, produce comparison table
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from datetime import datetime
from pathlib import Path

from tqdm import tqdm

from src.config import settings
from src.models import EvaluationResult, PromptComparisonRow, SceneDescription

logger = logging.getLogger(__name__)


def setup_logging(verbose: bool = False) -> None:
    """Configure logging."""
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s | %(levelname)-7s | %(name)s | %(message)s",
        datefmt="%H:%M:%S",
    )


def get_run_dir(run_id: str | None = None) -> Path:
    """Create and return a directory for this run's outputs."""
    if run_id is None:
        run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = settings.output_dir / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir


def load_checkpoint(run_dir: Path) -> set[str]:
    """Load set of already-processed image names for resumability."""
    checkpoint_file = run_dir / "checkpoint.json"
    if checkpoint_file.exists():
        with open(checkpoint_file, "r") as f:
            return set(json.load(f))
    return set()


def save_checkpoint(run_dir: Path, processed: set[str]) -> None:
    """Save checkpoint of processed images."""
    checkpoint_file = run_dir / "checkpoint.json"
    with open(checkpoint_file, "w") as f:
        json.dump(sorted(processed), f)


# ── Generate Mode ─────────────────────────────────────────────────────────────


def run_generate(
    prompt_id: str,
    limit: int | None = None,
    run_dir: Path | None = None,
    dry_run: bool = False,
) -> Path:
    """
    Generate scene descriptions for sampled images using a specific prompt.

    Args:
        prompt_id: ID of the prompt variant to use.
        limit: Max number of images to process.
        run_dir: Output directory (auto-created if None).
        dry_run: If True, print prompt and paths without making API calls.

    Returns:
        Path to the run directory with results.
    """
    from src.data.ground_truth import load_ground_truths
    from src.prompts.templates import get_prompt, get_prompt_info
    from src.vlm.client import GeminiClient

    prompt_text = get_prompt(prompt_id)
    prompt_info = get_prompt_info(prompt_id)

    if run_dir is None:
        run_dir = get_run_dir()

    logger.info(f"Prompt: {prompt_id} ({prompt_info['strategy']})")
    logger.info(f"Output: {run_dir}")

    if dry_run:
        logger.info("=== DRY RUN ===")
        logger.info(f"Prompt text:\n{prompt_text[:500]}...")
        return run_dir

    # Load data
    gts = load_ground_truths()
    image_names = sorted(gts.keys())
    if limit:
        image_names = image_names[:limit]

    # Resume support
    processed = load_checkpoint(run_dir)
    remaining = [n for n in image_names if n not in processed]
    logger.info(f"Total: {len(image_names)}, Already done: {len(processed)}, Remaining: {len(remaining)}")

    # Initialize client
    client = GeminiClient()
    results = []

    # Load existing results if any
    results_file = run_dir / f"results_{prompt_id}.json"
    if results_file.exists():
        with open(results_file, "r") as f:
            results = json.load(f)

    images_dir = settings.sampled_dir / "images"

    for image_name in tqdm(remaining, desc=f"Generating ({prompt_id})"):
        image_path = images_dir / image_name

        if not image_path.exists():
            logger.warning(f"Image not found: {image_path}")
            continue

        description = client.generate_scene_description(image_path, prompt_text)

        if description:
            result = {
                "image_name": image_name,
                "prompt_id": prompt_id,
                "description": description.model_dump(),
                "timestamp": datetime.now().isoformat(),
            }
            results.append(result)
        else:
            result = {
                "image_name": image_name,
                "prompt_id": prompt_id,
                "description": None,
                "error": "Failed to generate description",
                "timestamp": datetime.now().isoformat(),
            }
            results.append(result)

        processed.add(image_name)

        # Save incrementally
        with open(results_file, "w") as f:
            json.dump(results, f, indent=2)
        save_checkpoint(run_dir, processed)

    logger.info(f"Generated {len(results)} descriptions → {results_file}")
    logger.info(f"Remaining API quota: ~{client.requests_remaining} requests today")
    return run_dir


# ── Evaluate Mode ─────────────────────────────────────────────────────────────


def run_evaluate(
    results_file: Path,
    run_dir: Path | None = None,
    use_judge: bool = False,
) -> list[EvaluationResult]:
    """
    Evaluate generated descriptions against ground truth.

    Args:
        results_file: Path to the generated results JSON.
        run_dir: Output directory for evaluation results.
        use_judge: Whether to run LLM-as-judge evaluation (uses API quota).

    Returns:
        List of EvaluationResult objects.
    """
    from src.data.ground_truth import load_ground_truths
    from src.evaluation.bertscore import compute_bertscore
    from src.evaluation.completeness import compute_completeness
    from src.evaluation.hallucination import compute_hallucination

    if run_dir is None:
        run_dir = results_file.parent

    # Load results and ground truth
    with open(results_file, "r") as f:
        results = json.load(f)

    gts = load_ground_truths()
    evaluations = []

    # Collect texts for batch BERTScore
    predictions_text = []
    references_text = []
    valid_indices = []

    for i, result in enumerate(results):
        if result.get("description") is None:
            continue
        image_name = result["image_name"]
        if image_name not in gts:
            continue

        desc = SceneDescription.model_validate(result["description"])
        gt = gts[image_name]

        predictions_text.append(desc.summary)
        references_text.append(gt.description)
        valid_indices.append(i)

    # Batch BERTScore computation
    logger.info(f"Computing BERTScore for {len(predictions_text)} descriptions...")
    from src.evaluation.bertscore import compute_bertscore_batch
    bert_scores = compute_bertscore_batch(predictions_text, references_text)

    # Per-image evaluation
    judge_client = None
    if use_judge:
        from src.vlm.client import GeminiClient
        judge_client = GeminiClient()

    for batch_idx, result_idx in enumerate(tqdm(valid_indices, desc="Evaluating")):
        result = results[result_idx]
        image_name = result["image_name"]
        prompt_id = result["prompt_id"]
        desc = SceneDescription.model_validate(result["description"])
        gt = gts[image_name]

        # BERTScore (already computed)
        bs = bert_scores[batch_idx]

        # Hallucination
        hall = compute_hallucination(desc.objects, gt.objects)

        # Completeness
        comp = compute_completeness(desc)

        # Judge (optional)
        judge_score = None
        judge_reasoning = ""
        if use_judge and judge_client:
            from src.evaluation.judge import evaluate_with_judge
            judge_result = evaluate_with_judge(
                prediction_text=desc.model_dump_json(),
                ground_truth_text=gt.description,
                client=judge_client,
            )
            judge_score = judge_result.get("overall", 0)
            judge_reasoning = judge_result.get("reasoning", "")

        # Attribute accuracy
        eval_result = EvaluationResult(
            image_name=image_name,
            prompt_id=prompt_id,
            bert_score_precision=bs["precision"],
            bert_score_recall=bs["recall"],
            bert_score_f1=bs["f1"],
            hallucination_rate=hall.hallucination_rate,
            false_positive_objects=hall.false_positives,
            false_negative_objects=hall.false_negatives,
            completeness_score=comp["total"],
            judge_score=judge_score,
            judge_reasoning=judge_reasoning,
        )
        evaluations.append(eval_result)

    # Save evaluation results
    eval_file = run_dir / f"evaluation_{results_file.stem}.json"
    with open(eval_file, "w") as f:
        json.dump([e.model_dump() for e in evaluations], f, indent=2)

    # Print summary
    if evaluations:
        avg_bert = sum(e.bert_score_f1 for e in evaluations) / len(evaluations)
        avg_hall = sum(e.hallucination_rate for e in evaluations) / len(evaluations)
        avg_comp = sum(e.completeness_score for e in evaluations) / len(evaluations)
        print(f"\n{'='*60}")
        print(f"Evaluation Summary ({len(evaluations)} images)")
        print(f"{'='*60}")
        print(f"  Avg BERTScore F1:       {avg_bert:.4f}")
        print(f"  Avg Hallucination Rate:  {avg_hall:.4f}")
        print(f"  Avg Completeness:        {avg_comp:.4f}")
        if use_judge:
            avg_judge = sum(e.judge_score for e in evaluations if e.judge_score) / max(
                len([e for e in evaluations if e.judge_score]), 1
            )
            print(f"  Avg Judge Score:         {avg_judge:.2f}/5")
        print(f"{'='*60}\n")

    logger.info(f"Evaluation results saved to {eval_file}")
    return evaluations


# ── Compare Mode ──────────────────────────────────────────────────────────────


def run_compare(
    limit: int | None = None,
    prompt_ids: list[str] | None = None,
    use_judge: bool = False,
) -> Path:
    """
    Run all prompt variants on a subset and produce a comparison table.

    Args:
        limit: Number of images per prompt variant (default: eval_subset_size).
        prompt_ids: Specific prompt IDs to compare (default: all).
        use_judge: Whether to include LLM-as-judge scores.

    Returns:
        Path to the comparison run directory.
    """
    from src.prompts.registry import registry

    limit = limit or settings.eval_subset_size
    run_dir = get_run_dir(f"compare_{datetime.now().strftime('%Y%m%d_%H%M%S')}")

    variants = prompt_ids or registry.list_ids()
    logger.info(f"Comparing {len(variants)} prompt variants on {limit} images")

    comparison_rows = []

    for variant_id in variants:
        logger.info(f"\n{'='*60}")
        logger.info(f"Running prompt variant: {variant_id}")
        logger.info(f"{'='*60}")

        # Generate
        variant_dir = run_dir / variant_id
        variant_dir.mkdir(parents=True, exist_ok=True)
        run_generate(variant_id, limit=limit, run_dir=variant_dir)

        # Evaluate
        results_file = variant_dir / f"results_{variant_id}.json"
        if results_file.exists():
            evaluations = run_evaluate(results_file, run_dir=variant_dir, use_judge=use_judge)

            if evaluations:
                variant_info = registry.get(variant_id)
                row = PromptComparisonRow(
                    prompt_id=variant_id,
                    prompt_strategy=variant_info.strategy,
                    n_images=len(evaluations),
                    avg_bert_f1=round(
                        sum(e.bert_score_f1 for e in evaluations) / len(evaluations), 4
                    ),
                    avg_hallucination_rate=round(
                        sum(e.hallucination_rate for e in evaluations) / len(evaluations), 4
                    ),
                    avg_completeness=round(
                        sum(e.completeness_score for e in evaluations) / len(evaluations), 4
                    ),
                    avg_judge_score=(
                        round(
                            sum(e.judge_score for e in evaluations if e.judge_score)
                            / max(len([e for e in evaluations if e.judge_score]), 1),
                            2,
                        )
                        if use_judge
                        else None
                    ),
                )
                comparison_rows.append(row)

    # Save and display comparison table
    _save_comparison_table(comparison_rows, run_dir)
    return run_dir


def _save_comparison_table(rows: list[PromptComparisonRow], run_dir: Path) -> None:
    """Save and display the prompt comparison table."""
    import pandas as pd

    if not rows:
        logger.warning("No comparison results to display")
        return

    data = [r.model_dump() for r in rows]
    df = pd.DataFrame(data)

    # Sort by BERTScore F1 descending
    df = df.sort_values("avg_bert_f1", ascending=False)

    # Display
    print(f"\n{'='*80}")
    print("PROMPT VARIANT COMPARISON RESULTS")
    print(f"{'='*80}")
    print(df.to_string(index=False))
    print(f"{'='*80}")

    # Identify best variant
    best = df.iloc[0]
    baseline = df[df["prompt_id"] == "v1_baseline"]
    if not baseline.empty:
        improvement = (
            (best["avg_bert_f1"] - baseline.iloc[0]["avg_bert_f1"])
            / max(baseline.iloc[0]["avg_bert_f1"], 0.001)
            * 100
        )
        print(f"\nBest variant: {best['prompt_id']} ({best['prompt_strategy']})")
        print(f"Improvement over baseline: {improvement:+.1f}% BERTScore F1")

    # Save to CSV and JSON
    df.to_csv(run_dir / "comparison_table.csv", index=False)
    with open(run_dir / "comparison_table.json", "w") as f:
        json.dump(data, f, indent=2)

    logger.info(f"Comparison table saved to {run_dir}")


# ── CLI Entry Point ───────────────────────────────────────────────────────────


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

    # Compare
    comp = subparsers.add_parser("compare", help="Compare all prompt variants")
    comp.add_argument("--limit", type=int, default=None, help="Images per variant")
    comp.add_argument("--prompts", nargs="*", default=None, help="Specific prompt IDs")
    comp.add_argument("--judge", action="store_true", help="Include LLM-as-judge")

    # List prompts
    subparsers.add_parser("list-prompts", help="List available prompt variants")

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

    elif args.command == "list-prompts":
        from src.prompts.templates import list_prompts
        prompts = list_prompts()
        print(f"\n{'ID':<20} {'Strategy':<30} {'Description'}")
        print("-" * 80)
        for p in prompts:
            print(f"{p['id']:<20} {p['strategy']:<30} {p['description']}")
        print()

    else:
        parser.print_help()


if __name__ == "__main__":
    main()
