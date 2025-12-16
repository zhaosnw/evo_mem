"""
Evo-Memory: Benchmarking LLM Agent Test-time Learning with Self-Evolving Memory

A comprehensive streaming benchmark and framework for evaluating self-evolving
memory in LLM agents.

Key components:
- Memory: Core memory storage and retrieval
- Agents: Various agent implementations (ExpRAG, ReMem, baselines)
- Datasets: Single-turn and multi-turn task loaders
- Evaluation: Metrics and evaluation pipeline
- Experiments: Experiment runner and configuration
"""

__version__ = "0.1.0"

# Config
from .config import (
    Config,
    AgentType,
    DatasetType,
    LLMBackend,
)

# Memory
from .memory import (
    Memory,
    MemoryEntry,
    Retriever,
    EmbeddingRetriever,
    RecencyRetriever,
    ContextBuilder,
)

# Agents
from .agents import (
    BaseAgent,
    ExpRAGAgent,
    ExpRecentAgent,
    ReMemAgent,
    ReActAgent,
    AmemAgent,
    SelfRAGAgent,
    Mem0Agent,
    LangMemAgent,
    DynamicCheatsheetAgent,
    AWMAgent,
)

# Datasets
from .datasets import (
    BaseDataset,
    TaskInstance,
    DatasetSplit,
    # Single-turn
    MMLUProDataset,
    GPQADataset,
    AIMEDataset,
    ToolBenchDataset,
    # Multi-turn
    AlfWorldDataset,
    BabyAIDataset,
    PDDLDataset,
    ScienceWorldDataset,
)

# Evaluation
from .evaluation import (
    Evaluator,
    EvaluationConfig,
    StreamResult,
    AnswerAccuracy,
    SuccessRate,
    ProgressRate,
)

# Experiments
from .experiments import (
    ExperimentConfig,
    ExperimentRunner,
    run_quick_experiment,
)

__all__ = [
    # Version
    "__version__",
    # Config
    "Config",
    "AgentType",
    "DatasetType",
    "LLMBackend",
    # Memory
    "Memory",
    "MemoryEntry",
    "Retriever",
    "EmbeddingRetriever",
    "RecencyRetriever",
    "ContextBuilder",
    # Agents
    "BaseAgent",
    "ExpRAGAgent",
    "ExpRecentAgent",
    "ReMemAgent",
    "ReActAgent",
    "AmemAgent",
    "SelfRAGAgent",
    "Mem0Agent",
    "LangMemAgent",
    "DynamicCheatsheetAgent",
    "AWMAgent",
    # Datasets
    "BaseDataset",
    "TaskInstance",
    "DatasetSplit",
    "MMLUProDataset",
    "GPQADataset",
    "AIMEDataset",
    "ToolBenchDataset",
    "AlfWorldDataset",
    "BabyAIDataset",
    "PDDLDataset",
    "ScienceWorldDataset",
    # Evaluation
    "Evaluator",
    "EvaluationConfig",
    "StreamResult",
    "AnswerAccuracy",
    "SuccessRate",
    "ProgressRate",
    # Experiments
    "ExperimentConfig",
    "ExperimentRunner",
    "run_quick_experiment",
]
