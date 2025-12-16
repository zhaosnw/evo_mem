"""Configuration classes for Evo-Memory experiments."""

from dataclasses import dataclass, field
from typing import Optional, List, Dict, Any
from enum import Enum
import json
import yaml
from pathlib import Path


class LLMBackend(Enum):
    """Supported LLM backends."""
    OPENAI = "openai"
    ANTHROPIC = "anthropic"
    GOOGLE = "google"
    LOCAL = "local"


class AgentType(Enum):
    """Supported agent types."""
    BASELINE = "baseline"
    HISTORY = "history"
    REACT = "react"
    AMEM = "amem"
    SELFRAG = "selfrag"
    MEMOS = "memos"
    MEM0 = "mem0"
    LANGMEM = "langmem"
    DC_CU = "dc_cu"  # Dynamic Cheatsheet Cumulative
    DC_RS = "dc_rs"  # Dynamic Cheatsheet Retrieval Synthesis
    AWM = "awm"      # Agent Workflow Memory
    EXPRECENT = "exprecent"
    EXPRAG = "exprag"
    REMEM = "remem"


class DatasetType(Enum):
    """Supported dataset types."""
    # Single-turn datasets
    MMLU_PRO = "mmlu_pro"
    GPQA = "gpqa"
    AIME_24 = "aime_24"
    AIME_25 = "aime_25"
    TOOLBENCH = "toolbench"
    # Multi-turn datasets
    ALFWORLD = "alfworld"
    BABYAI = "babyai"
    PDDL = "pddl"
    SCIENCEWORLD = "scienceworld"
    JERICHO = "jericho"


@dataclass
class ModelConfig:
    """Configuration for LLM model."""
    backend: LLMBackend = LLMBackend.OPENAI
    model_name: str = "gpt-4"
    api_key: Optional[str] = None
    api_base: Optional[str] = None
    temperature: float = 0.0
    max_tokens: int = 4096
    top_p: float = 1.0
    timeout: int = 120

    def to_dict(self) -> Dict[str, Any]:
        return {
            "backend": self.backend.value,
            "model_name": self.model_name,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
            "top_p": self.top_p,
            "timeout": self.timeout,
        }


@dataclass
class MemoryConfig:
    """Configuration for memory module."""
    # Retriever settings
    retriever_model: str = "BAAI/bge-base-en-v1.5"
    embedding_dim: int = 768
    top_k: int = 4
    similarity_threshold: float = 0.0

    # Memory capacity
    max_memory_size: int = 1000
    enable_pruning: bool = True
    pruning_threshold: float = 0.3

    # Experience storage
    store_successful_only: bool = True
    include_trajectory: bool = True
    include_feedback: bool = True

    def to_dict(self) -> Dict[str, Any]:
        return {
            "retriever_model": self.retriever_model,
            "embedding_dim": self.embedding_dim,
            "top_k": self.top_k,
            "similarity_threshold": self.similarity_threshold,
            "max_memory_size": self.max_memory_size,
            "enable_pruning": self.enable_pruning,
            "pruning_threshold": self.pruning_threshold,
            "store_successful_only": self.store_successful_only,
            "include_trajectory": self.include_trajectory,
            "include_feedback": self.include_feedback,
        }


@dataclass
class ExperimentConfig:
    """Configuration for experiments."""
    # Experiment identification
    name: str = "evo_memory_experiment"
    seed: int = 42

    # Dataset settings
    dataset: DatasetType = DatasetType.ALFWORLD
    dataset_path: Optional[str] = None
    dataset_split: str = "test"
    num_samples: Optional[int] = None

    # Agent settings
    agent_type: AgentType = AgentType.REMEM
    max_steps: int = 50  # Max steps per task for multi-turn
    max_iterations: int = 10  # Max Think/Refine iterations per step

    # Evaluation
    eval_batch_size: int = 1
    save_trajectories: bool = True

    # Output
    output_dir: str = "./outputs"
    log_level: str = "INFO"

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "seed": self.seed,
            "dataset": self.dataset.value,
            "dataset_path": self.dataset_path,
            "dataset_split": self.dataset_split,
            "num_samples": self.num_samples,
            "agent_type": self.agent_type.value,
            "max_steps": self.max_steps,
            "max_iterations": self.max_iterations,
            "eval_batch_size": self.eval_batch_size,
            "save_trajectories": self.save_trajectories,
            "output_dir": self.output_dir,
            "log_level": self.log_level,
        }


@dataclass
class Config:
    """Main configuration combining all sub-configs."""
    model: ModelConfig = field(default_factory=ModelConfig)
    memory: MemoryConfig = field(default_factory=MemoryConfig)
    experiment: ExperimentConfig = field(default_factory=ExperimentConfig)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "model": self.model.to_dict(),
            "memory": self.memory.to_dict(),
            "experiment": self.experiment.to_dict(),
        }

    def save(self, path: str) -> None:
        """Save config to file (JSON or YAML)."""
        path = Path(path)
        data = self.to_dict()

        if path.suffix == ".json":
            with open(path, "w") as f:
                json.dump(data, f, indent=2)
        elif path.suffix in [".yaml", ".yml"]:
            with open(path, "w") as f:
                yaml.dump(data, f, default_flow_style=False)
        else:
            raise ValueError(f"Unsupported config format: {path.suffix}")

    @classmethod
    def load(cls, path: str) -> "Config":
        """Load config from file."""
        path = Path(path)

        if path.suffix == ".json":
            with open(path) as f:
                data = json.load(f)
        elif path.suffix in [".yaml", ".yml"]:
            with open(path) as f:
                data = yaml.safe_load(f)
        else:
            raise ValueError(f"Unsupported config format: {path.suffix}")

        return cls.from_dict(data)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Config":
        """Create config from dictionary."""
        model_data = data.get("model", {})
        memory_data = data.get("memory", {})
        experiment_data = data.get("experiment", {})

        # Convert string enums back to enum types
        if "backend" in model_data:
            model_data["backend"] = LLMBackend(model_data["backend"])
        if "dataset" in experiment_data:
            experiment_data["dataset"] = DatasetType(experiment_data["dataset"])
        if "agent_type" in experiment_data:
            experiment_data["agent_type"] = AgentType(experiment_data["agent_type"])

        return cls(
            model=ModelConfig(**model_data),
            memory=MemoryConfig(**memory_data),
            experiment=ExperimentConfig(**experiment_data),
        )
