"""Main entry point for Evo-Memory benchmark.

Usage:
    python -m evo_memory.main run --agent exprag --dataset mmlu_pro
    python -m evo_memory.main run --config experiments/config.yaml
    python -m evo_memory.main list-agents
    python -m evo_memory.main list-datasets
"""

import argparse
import logging
import sys
from pathlib import Path

from .config import AgentType, DatasetType, LLMBackend
from .experiments import (
    ExperimentConfig,
    ExperimentRunner,
    BatchExperimentRunner,
    run_experiment_from_config,
    run_quick_experiment,
    AGENT_REGISTRY,
    DATASET_REGISTRY,
)

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def setup_parser() -> argparse.ArgumentParser:
    """Set up argument parser."""
    parser = argparse.ArgumentParser(
        description="Evo-Memory: Benchmarking LLM Agent Test-time Learning",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # Run command
    run_parser = subparsers.add_parser("run", help="Run an experiment")
    run_parser.add_argument(
        "--config",
        type=str,
        help="Path to YAML config file",
    )
    run_parser.add_argument(
        "--agent",
        type=str,
        choices=[a.value for a in AgentType],
        help="Agent type",
    )
    run_parser.add_argument(
        "--dataset",
        type=str,
        choices=[d.value for d in DatasetType],
        help="Dataset type",
    )
    run_parser.add_argument(
        "--model",
        type=str,
        default="gpt-4o-mini",
        help="Model name (default: gpt-4o-mini)",
    )
    run_parser.add_argument(
        "--backend",
        type=str,
        choices=[b.value for b in LLMBackend],
        default="openai",
        help="LLM backend (default: openai)",
    )
    run_parser.add_argument(
        "--task-limit",
        type=int,
        default=None,
        help="Maximum number of tasks to evaluate",
    )
    run_parser.add_argument(
        "--num-streams",
        type=int,
        default=1,
        help="Number of evaluation streams (default: 1)",
    )
    run_parser.add_argument(
        "--output-dir",
        type=str,
        default="./results",
        help="Output directory (default: ./results)",
    )
    run_parser.add_argument(
        "--memory-size",
        type=int,
        default=1000,
        help="Memory size limit (default: 1000)",
    )
    run_parser.add_argument(
        "--retrieval-k",
        type=int,
        default=4,
        help="Number of items to retrieve (default: 4)",
    )
    run_parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed (default: 42)",
    )
    run_parser.add_argument(
        "--quiet",
        action="store_true",
        help="Reduce output verbosity",
    )

    # Batch command
    batch_parser = subparsers.add_parser("batch", help="Run batch experiments")
    batch_parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to base YAML config file",
    )
    batch_parser.add_argument(
        "--agents",
        nargs="+",
        type=str,
        help="Agent types to compare",
    )
    batch_parser.add_argument(
        "--datasets",
        nargs="+",
        type=str,
        help="Dataset types to compare",
    )
    batch_parser.add_argument(
        "--output-dir",
        type=str,
        default="./batch_results",
        help="Output directory",
    )

    # List agents command
    list_agents_parser = subparsers.add_parser("list-agents", help="List available agents")

    # List datasets command
    list_datasets_parser = subparsers.add_parser("list-datasets", help="List available datasets")

    # Quick run command
    quick_parser = subparsers.add_parser("quick", help="Run a quick test experiment")
    quick_parser.add_argument(
        "--agent",
        type=str,
        default="exprag",
        help="Agent type (default: exprag)",
    )
    quick_parser.add_argument(
        "--dataset",
        type=str,
        default="mmlu_pro",
        help="Dataset type (default: mmlu_pro)",
    )
    quick_parser.add_argument(
        "--tasks",
        type=int,
        default=10,
        help="Number of tasks (default: 10)",
    )

    return parser


def run_command(args):
    """Execute run command."""
    if args.config:
        # Run from config file
        logger.info(f"Running experiment from config: {args.config}")
        result = run_experiment_from_config(args.config)
    else:
        # Check required arguments
        if not args.agent or not args.dataset:
            logger.error("Either --config or both --agent and --dataset are required")
            sys.exit(1)

        # Create config from CLI arguments
        config = ExperimentConfig(
            name=f"{args.agent}_{args.dataset}",
            agent_type=AgentType(args.agent),
            dataset_type=DatasetType(args.dataset),
            llm_backend=LLMBackend(args.backend),
            model_name=args.model,
            memory_size=args.memory_size,
            retrieval_k=args.retrieval_k,
            num_streams=args.num_streams,
            task_limit=args.task_limit,
            output_dir=args.output_dir,
            seed=args.seed,
            verbose=not args.quiet,
        )

        runner = ExperimentRunner(config)
        result = runner.run()

    # Print summary
    print("\n" + "=" * 60)
    print("EXPERIMENT RESULTS")
    print("=" * 60)

    for key, value in result.items():
        if key not in ["config", "results"] and not key.endswith("_values"):
            if isinstance(value, float):
                print(f"{key}: {value:.4f}")
            else:
                print(f"{key}: {value}")

    return result


def batch_command(args):
    """Execute batch command."""
    base_config = ExperimentConfig.from_yaml(args.config)

    variations = {}
    if args.agents:
        variations["agent_type"] = [AgentType(a) for a in args.agents]
    if args.datasets:
        variations["dataset_type"] = [DatasetType(d) for d in args.datasets]

    if not variations:
        logger.error("At least one of --agents or --datasets is required")
        sys.exit(1)

    base_config.output_dir = args.output_dir

    runner = BatchExperimentRunner(base_config, variations)
    results = runner.run()

    print("\n" + "=" * 60)
    print("BATCH EXPERIMENT SUMMARY")
    print("=" * 60)
    print(f"Total experiments: {len(results)}")

    for result in results:
        name = result.get("config", {}).get("name", "unknown")
        acc_mean = result.get("accuracy_mean", 0)
        acc_std = result.get("accuracy_std", 0)
        print(f"{name}: {acc_mean:.4f} ± {acc_std:.4f}")

    return results


def list_agents_command(args):
    """List available agents."""
    print("\nAvailable Agents:")
    print("-" * 40)
    for agent_type in AgentType:
        cls = AGENT_REGISTRY.get(agent_type)
        cls_name = cls.__name__ if cls else "Not implemented"
        print(f"  {agent_type.value:<25} ({cls_name})")

    print("\nAgent Descriptions:")
    print("-" * 40)
    descriptions = {
        "exprag": "Experience retrieval-augmented generation (baseline)",
        "exp_recent": "Recency-based experience retrieval",
        "remem": "ReMem with Think-Act-Refine loop",
        "react": "ReAct reasoning and acting",
        "amem": "A-mem with experience accumulation",
        "selfrag": "Self-RAG with reflection",
        "mem0": "Mem0 memory system",
        "langmem": "LangMem hierarchical memory",
        "dynamic_cheatsheet": "Dynamic cheatsheet approach",
        "awm": "AWM workflow memory",
    }
    for name, desc in descriptions.items():
        print(f"  {name:<25} {desc}")


def list_datasets_command(args):
    """List available datasets."""
    print("\nAvailable Datasets:")
    print("-" * 40)

    print("\nSingle-Turn Tasks:")
    single_turn = ["mmlu_pro", "gpqa", "aime24", "aime25", "toolbench"]
    for d in single_turn:
        cls = DATASET_REGISTRY.get(DatasetType(d))
        cls_name = cls.__name__ if cls else "Not implemented"
        print(f"  {d:<25} ({cls_name})")

    print("\nMulti-Turn Tasks:")
    multi_turn = ["alfworld", "babyai", "pddl", "scienceworld"]
    for d in multi_turn:
        cls = DATASET_REGISTRY.get(DatasetType(d))
        cls_name = cls.__name__ if cls else "Not implemented"
        print(f"  {d:<25} ({cls_name})")

    print("\nDataset Descriptions:")
    print("-" * 40)
    descriptions = {
        "mmlu_pro": "MMLU-Pro multiple choice questions",
        "gpqa": "Graduate-level science questions",
        "aime24": "AIME 2024 math competition",
        "aime25": "AIME 2025 math competition",
        "toolbench": "API tool calling tasks",
        "alfworld": "Household instruction following",
        "babyai": "Grid world navigation",
        "pddl": "Symbolic planning (Blocksworld)",
        "scienceworld": "Science experimentation",
    }
    for name, desc in descriptions.items():
        print(f"  {name:<25} {desc}")


def quick_command(args):
    """Run a quick test experiment."""
    logger.info(f"Running quick experiment: {args.agent} on {args.dataset}")

    result = run_quick_experiment(
        agent_type=args.agent,
        dataset_type=args.dataset,
        task_limit=args.tasks,
    )

    print("\n" + "=" * 60)
    print("QUICK EXPERIMENT RESULTS")
    print("=" * 60)
    print(f"Agent: {args.agent}")
    print(f"Dataset: {args.dataset}")
    print(f"Tasks: {args.tasks}")
    print(f"Accuracy: {result.get('accuracy_mean', 0):.4f}")

    return result


def main():
    """Main entry point."""
    parser = setup_parser()
    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        sys.exit(0)

    try:
        if args.command == "run":
            run_command(args)
        elif args.command == "batch":
            batch_command(args)
        elif args.command == "list-agents":
            list_agents_command(args)
        elif args.command == "list-datasets":
            list_datasets_command(args)
        elif args.command == "quick":
            quick_command(args)
        else:
            parser.print_help()
            sys.exit(1)
    except KeyboardInterrupt:
        logger.info("\nExperiment interrupted by user")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Experiment failed: {e}")
        raise


if __name__ == "__main__":
    main()
