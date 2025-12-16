# Evo-Memory: Benchmarking LLM Agent Test-time Learning with Self-Evolving Memory

A comprehensive streaming benchmark and framework for evaluating self-evolving memory in LLM agents.

## Overview

Evo-Memory provides:
- **Unified Memory Framework**: A formalization of memory-augmented agents as (F, U, R, C) tuples
- **Search-Synthesize-Evolve Loop**: Core mechanism for test-time learning
- **Multiple Agent Implementations**: ExpRAG, ReMem, and various baselines
- **Comprehensive Datasets**: Single-turn (MMLU-Pro, GPQA, AIME, ToolBench) and Multi-turn (AlfWorld, BabyAI, PDDL, ScienceWorld)
- **Evaluation Pipeline**: Metrics for accuracy, success rate, progress rate, and step efficiency

## Installation

```bash
pip install -e .
```

Or install with all optional dependencies:

```bash
pip install -e ".[all]"
```

## Quick Start

### Run a quick experiment

```bash
python -m evo_memory quick --agent exprag --dataset mmlu_pro --tasks 10
```

### Run with configuration

```bash
python -m evo_memory run --agent remem --dataset mmlu_pro --model gpt-4o-mini
```

### List available agents and datasets

```bash
python -m evo_memory list-agents
python -m evo_memory list-datasets
```

## Architecture

### Memory-Augmented Agent

The framework formalizes memory-augmented agents as a tuple (F, U, R, C):
- **F**: Base LLM for generation
- **U**: Memory update function
- **R**: Retrieval function
- **C**: Context construction function

### Search-Synthesize-Evolve Loop

At each task t:
1. **Search**: R_t = R(M_t, x_t) - Retrieve relevant memories
2. **Synthesize**: C̃_t = C(x_t, R_t) - Construct context
3. **Evolve**: M_{t+1} = U(M_t, m_t) - Update memory

## Agents

| Agent | Description |
|-------|-------------|
| ExpRAG | Experience retrieval-augmented generation |
| ExpRecent | Recency-based experience retrieval |
| ReMem | Think-Act-Refine loop for continual improvement |
| ReAct | Reasoning and acting with memory |
| A-mem | Experience accumulation |
| Self-RAG | Self-reflection with retrieval |
| Mem0 | Hierarchical memory system |
| LangMem | Language-based memory management |
| DynamicCheatsheet | Dynamic knowledge aggregation |
| AWM | Agent workflow memory |

## Datasets

### Single-Turn
- **MMLU-Pro**: Multiple choice questions
- **GPQA**: Graduate-level science questions
- **AIME**: Math competition problems
- **ToolBench**: API calling tasks

### Multi-Turn
- **AlfWorld**: Household instruction following
- **BabyAI**: Grid world navigation
- **PDDL**: Symbolic planning
- **ScienceWorld**: Science experiments

## Usage

### Python API

```python
from evo_memory import (
    ExperimentConfig,
    ExperimentRunner,
    AgentType,
    DatasetType,
)

# Create configuration
config = ExperimentConfig(
    name="my_experiment",
    agent_type=AgentType.EXPRAG,
    dataset_type=DatasetType.MMLU_PRO,
    model_name="gpt-4o-mini",
    num_streams=3,
)

# Run experiment
runner = ExperimentRunner(config)
results = runner.run()

print(f"Accuracy: {results['accuracy_mean']:.4f}")
```

### Configuration File

Create a YAML config file:

```yaml
name: my_experiment
agent_type: exprag
dataset_type: mmlu_pro
llm_backend: openai
model_name: gpt-4o-mini
memory_size: 1000
retrieval_k: 4
num_streams: 3
output_dir: ./results
```

Run with:

```bash
python -m evo_memory run --config config.yaml
```

## Evaluation Metrics

- **Answer Accuracy**: Correctness for single-turn tasks
- **Success Rate**: Task completion for multi-turn tasks
- **Progress Rate**: Partial progress measurement
- **Step Efficiency**: Efficiency in multi-turn tasks

## License

MIT License
