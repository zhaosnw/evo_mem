"""Experiment runner for Evo-Memory benchmark.

Provides a unified interface for running experiments across
different agents, datasets, and configurations.
"""

from typing import List, Dict, Any, Optional, Type
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
import json
import logging
import os
import yaml

from ..config import Config, AgentType, DatasetType, LLMBackend
from ..agents.base import BaseAgent
from ..agents.exprag import ExpRAGAgent, ExpRecentAgent
from ..agents.remem import ReMemAgent
from ..agents.react import ReActAgent
from ..agents.amem import AmemAgent
from ..agents.selfrag import SelfRAGAgent
from ..agents.mem0 import Mem0Agent
from ..agents.langmem import LangMemAgent
from ..agents.dynamic_cheatsheet import DynamicCheatsheetAgent
from ..agents.awm import AWMAgent
from ..datasets.base import BaseDataset
from ..datasets.single_turn import (
    MMLUProDataset,
    GPQADataset,
    AIMEDataset,
    ToolBenchDataset,
)
from ..datasets.multi_turn import (
    AlfWorldDataset,
    BabyAIDataset,
    PDDLDataset,
    ScienceWorldDataset,
)
from ..evaluation import Evaluator, EvaluationConfig, StreamResult, save_results
from ..llm.base import BaseLLM
from ..llm.openai_llm import OpenAILLM
from ..llm.anthropic_llm import AnthropicLLM
from ..llm.google_llm import GoogleLLM
from ..memory.base import Memory
from ..memory.retriever import EmbeddingRetriever, RecencyRetriever

logger = logging.getLogger(__name__)


# Agent registry
AGENT_REGISTRY: Dict[AgentType, Type[BaseAgent]] = {
    AgentType.EXPRAG: ExpRAGAgent,
    AgentType.EXP_RECENT: ExpRecentAgent,
    AgentType.REMEM: ReMemAgent,
    AgentType.REACT: ReActAgent,
    AgentType.AMEM: AmemAgent,
    AgentType.SELFRAG: SelfRAGAgent,
    AgentType.MEM0: Mem0Agent,
    AgentType.LANGMEM: LangMemAgent,
    AgentType.DYNAMIC_CHEATSHEET: DynamicCheatsheetAgent,
    AgentType.AWM: AWMAgent,
}

# Dataset registry
DATASET_REGISTRY: Dict[DatasetType, Type[BaseDataset]] = {
    DatasetType.MMLU_PRO: MMLUProDataset,
    DatasetType.GPQA: GPQADataset,
    DatasetType.AIME24: AIMEDataset,
    DatasetType.AIME25: AIMEDataset,
    DatasetType.TOOLBENCH: ToolBenchDataset,
    DatasetType.ALFWORLD: AlfWorldDataset,
    DatasetType.BABYAI: BabyAIDataset,
    DatasetType.PDDL: PDDLDataset,
    DatasetType.SCIENCEWORLD: ScienceWorldDataset,
}

# LLM backend registry
LLM_REGISTRY: Dict[LLMBackend, Type[BaseLLM]] = {
    LLMBackend.OPENAI: OpenAILLM,
    LLMBackend.ANTHROPIC: AnthropicLLM,
    LLMBackend.GOOGLE: GoogleLLM,
}


@dataclass
class ExperimentConfig:
    """Configuration for a single experiment run."""
    name: str
    agent_type: AgentType
    dataset_type: DatasetType
    llm_backend: LLMBackend = LLMBackend.OPENAI
    model_name: str = "gpt-4o-mini"
    memory_size: int = 1000
    retrieval_k: int = 4
    num_streams: int = 1
    task_limit: Optional[int] = None
    output_dir: str = "./results"
    seed: int = 42

    # Dataset-specific settings
    dataset_split: str = "test"
    dataset_path: Optional[str] = None

    # Evaluation settings
    max_steps: int = 50
    save_trajectories: bool = True
    verbose: bool = True

    # Additional agent kwargs
    agent_kwargs: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "name": self.name,
            "agent_type": self.agent_type.value,
            "dataset_type": self.dataset_type.value,
            "llm_backend": self.llm_backend.value,
            "model_name": self.model_name,
            "memory_size": self.memory_size,
            "retrieval_k": self.retrieval_k,
            "num_streams": self.num_streams,
            "task_limit": self.task_limit,
            "output_dir": self.output_dir,
            "seed": self.seed,
            "dataset_split": self.dataset_split,
            "dataset_path": self.dataset_path,
            "max_steps": self.max_steps,
            "save_trajectories": self.save_trajectories,
            "verbose": self.verbose,
            "agent_kwargs": self.agent_kwargs,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ExperimentConfig":
        """Create from dictionary."""
        data = data.copy()
        data["agent_type"] = AgentType(data["agent_type"])
        data["dataset_type"] = DatasetType(data["dataset_type"])
        data["llm_backend"] = LLMBackend(data["llm_backend"])
        return cls(**data)

    @classmethod
    def from_yaml(cls, path: str) -> "ExperimentConfig":
        """Load from YAML file."""
        with open(path, 'r') as f:
            data = yaml.safe_load(f)
        return cls.from_dict(data)

    def save_yaml(self, path: str):
        """Save to YAML file."""
        with open(path, 'w') as f:
            yaml.dump(self.to_dict(), f, default_flow_style=False)


class ExperimentRunner:
    """
    Main experiment runner for Evo-Memory benchmark.

    Handles:
    - Agent and dataset instantiation
    - Running evaluations
    - Saving results
    """

    def __init__(self, config: ExperimentConfig):
        self.config = config
        self.results: List[StreamResult] = []

        # Set up output directory
        self.output_dir = Path(config.output_dir) / config.name
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Save config
        config.save_yaml(str(self.output_dir / "config.yaml"))

    def _create_llm(self) -> BaseLLM:
        """Create LLM instance."""
        llm_cls = LLM_REGISTRY.get(self.config.llm_backend)
        if not llm_cls:
            raise ValueError(f"Unknown LLM backend: {self.config.llm_backend}")

        return llm_cls(model_name=self.config.model_name)

    def _create_memory(self) -> Memory:
        """Create memory instance."""
        return Memory(max_size=self.config.memory_size)

    def _create_retriever(self):
        """Create retriever instance."""
        if self.config.agent_type == AgentType.EXP_RECENT:
            return RecencyRetriever(top_k=self.config.retrieval_k)
        else:
            return EmbeddingRetriever(top_k=self.config.retrieval_k)

    def _create_agent(self) -> BaseAgent:
        """Create agent instance."""
        agent_cls = AGENT_REGISTRY.get(self.config.agent_type)
        if not agent_cls:
            raise ValueError(f"Unknown agent type: {self.config.agent_type}")

        llm = self._create_llm()
        memory = self._create_memory()
        retriever = self._create_retriever()

        return agent_cls(
            llm=llm,
            memory=memory,
            retriever=retriever,
            **self.config.agent_kwargs,
        )

    def _create_dataset(self) -> BaseDataset:
        """Create dataset instance."""
        dataset_cls = DATASET_REGISTRY.get(self.config.dataset_type)
        if not dataset_cls:
            raise ValueError(f"Unknown dataset type: {self.config.dataset_type}")

        kwargs = {}
        if self.config.dataset_path:
            kwargs["data_path"] = self.config.dataset_path

        # Handle AIME year variants
        if self.config.dataset_type == DatasetType.AIME24:
            kwargs["year"] = 2024
        elif self.config.dataset_type == DatasetType.AIME25:
            kwargs["year"] = 2025

        return dataset_cls(**kwargs)

    def run(self) -> Dict[str, Any]:
        """
        Run the experiment.

        Returns:
            Dictionary with aggregated results
        """
        logger.info(f"Starting experiment: {self.config.name}")
        logger.info(f"Agent: {self.config.agent_type.value}")
        logger.info(f"Dataset: {self.config.dataset_type.value}")
        logger.info(f"Streams: {self.config.num_streams}")

        all_metrics = []

        for stream_idx in range(self.config.num_streams):
            logger.info(f"\n=== Stream {stream_idx + 1}/{self.config.num_streams} ===")

            # Create fresh instances
            agent = self._create_agent()
            dataset = self._create_dataset()

            # Create evaluator
            eval_config = EvaluationConfig(
                max_steps=self.config.max_steps,
                save_trajectories=self.config.save_trajectories,
                verbose=self.config.verbose,
            )
            evaluator = Evaluator(agent, dataset, eval_config)

            # Run evaluation
            result = evaluator.evaluate_stream(
                stream_id=f"stream_{stream_idx}",
                task_limit=self.config.task_limit,
            )

            self.results.append(result)
            all_metrics.append(result.final_metrics)

            # Save individual stream result
            save_results(
                result,
                str(self.output_dir / f"stream_{stream_idx}_results.json"),
            )

        # Aggregate results
        aggregated = self._aggregate_results(all_metrics)

        # Save aggregated results
        self._save_aggregated(aggregated)

        logger.info(f"\nExperiment completed: {self.config.name}")
        logger.info(f"Results saved to: {self.output_dir}")

        return aggregated

    def _aggregate_results(self, all_metrics: List[Dict[str, float]]) -> Dict[str, Any]:
        """Aggregate metrics across streams."""
        import numpy as np

        if not all_metrics:
            return {}

        aggregated = {
            "config": self.config.to_dict(),
            "num_streams": len(all_metrics),
            "timestamp": datetime.now().isoformat(),
        }

        # Collect all metric keys
        all_keys = set()
        for m in all_metrics:
            all_keys.update(m.keys())

        # Compute mean and std for each metric
        for key in all_keys:
            values = [m.get(key, 0) for m in all_metrics]
            aggregated[f"{key}_mean"] = float(np.mean(values))
            aggregated[f"{key}_std"] = float(np.std(values))
            aggregated[f"{key}_values"] = values

        return aggregated

    def _save_aggregated(self, aggregated: Dict[str, Any]):
        """Save aggregated results."""
        output_path = self.output_dir / "aggregated_results.json"
        with open(output_path, 'w') as f:
            json.dump(aggregated, f, indent=2)


class BatchExperimentRunner:
    """
    Runner for batch experiments across multiple configurations.

    Useful for comparing multiple agents or datasets.
    """

    def __init__(
        self,
        base_config: ExperimentConfig,
        variations: Dict[str, List[Any]],
    ):
        """
        Initialize batch runner.

        Args:
            base_config: Base experiment configuration
            variations: Dictionary of parameter names to values to vary
        """
        self.base_config = base_config
        self.variations = variations
        self.all_results: List[Dict[str, Any]] = []

    def generate_configs(self) -> List[ExperimentConfig]:
        """Generate all configuration combinations."""
        from itertools import product

        configs = []

        # Get all variation combinations
        keys = list(self.variations.keys())
        values = list(self.variations.values())

        for combo in product(*values):
            config_dict = self.base_config.to_dict()

            for key, value in zip(keys, combo):
                config_dict[key] = value

            # Update name to reflect variation
            var_str = "_".join(f"{k}={v}" for k, v in zip(keys, combo))
            config_dict["name"] = f"{self.base_config.name}_{var_str}"

            configs.append(ExperimentConfig.from_dict(config_dict))

        return configs

    def run(self) -> List[Dict[str, Any]]:
        """Run all experiments."""
        configs = self.generate_configs()
        logger.info(f"Running {len(configs)} experiment configurations")

        for idx, config in enumerate(configs):
            logger.info(f"\n=== Experiment {idx + 1}/{len(configs)}: {config.name} ===")

            runner = ExperimentRunner(config)
            result = runner.run()

            self.all_results.append(result)

        # Save summary
        self._save_summary()

        return self.all_results

    def _save_summary(self):
        """Save summary of all experiments."""
        summary = {
            "base_config": self.base_config.to_dict(),
            "variations": {k: [str(v) for v in vs] for k, vs in self.variations.items()},
            "num_experiments": len(self.all_results),
            "results": self.all_results,
        }

        output_dir = Path(self.base_config.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        with open(output_dir / "batch_summary.json", 'w') as f:
            json.dump(summary, f, indent=2)


def run_experiment_from_config(config_path: str) -> Dict[str, Any]:
    """
    Run experiment from config file.

    Args:
        config_path: Path to YAML config file

    Returns:
        Experiment results
    """
    config = ExperimentConfig.from_yaml(config_path)
    runner = ExperimentRunner(config)
    return runner.run()


def run_quick_experiment(
    agent_type: str = "exprag",
    dataset_type: str = "mmlu_pro",
    model_name: str = "gpt-4o-mini",
    task_limit: int = 10,
    output_dir: str = "./results",
) -> Dict[str, Any]:
    """
    Run a quick experiment with minimal configuration.

    Args:
        agent_type: Agent type name
        dataset_type: Dataset type name
        model_name: Model name
        task_limit: Maximum tasks to evaluate
        output_dir: Output directory

    Returns:
        Experiment results
    """
    config = ExperimentConfig(
        name=f"quick_{agent_type}_{dataset_type}",
        agent_type=AgentType(agent_type),
        dataset_type=DatasetType(dataset_type),
        model_name=model_name,
        task_limit=task_limit,
        output_dir=output_dir,
        num_streams=1,
        verbose=True,
    )

    runner = ExperimentRunner(config)
    return runner.run()
