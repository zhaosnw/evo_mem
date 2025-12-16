"""Evaluator for Evo-Memory benchmark.

Implements the main evaluation pipeline for both single-turn
and multi-turn tasks with streaming evaluation.
"""

from typing import List, Dict, Any, Optional, Generator, Tuple
from dataclasses import dataclass, field
from datetime import datetime
import json
import logging
import time

from .metrics import (
    BaseMetric,
    AnswerAccuracy,
    SuccessRate,
    ProgressRate,
    StepEfficiency,
    AIMEAccuracy,
    ToolBenchAccuracy,
    EvaluationResult,
    StreamEvaluationResult,
    StreamMetrics,
    MetricType,
)
from ..agents.base import BaseAgent
from ..datasets.base import BaseDataset, TaskInstance, DatasetSplit, MultiTurnDataset
from ..memory.base import Memory

logger = logging.getLogger(__name__)


@dataclass
class EvaluationConfig:
    """Configuration for evaluation."""
    max_steps: int = 50  # Max steps for multi-turn
    timeout: int = 300  # Timeout in seconds
    save_trajectories: bool = True
    save_memory_snapshots: bool = False
    verbose: bool = True
    stream_shuffle: bool = False  # Whether to shuffle task order
    num_streams: int = 1  # Number of evaluation streams


@dataclass
class TaskResult:
    """Result of a single task execution."""
    task_id: str
    input_text: str
    prediction: str
    target: str
    correct: bool
    score: float
    steps: int = 1
    trajectory: List[Dict] = field(default_factory=list)
    memory_used: int = 0
    time_taken: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class StreamResult:
    """Result of evaluating a task stream."""
    stream_id: str
    agent_name: str
    dataset_name: str
    task_results: List[TaskResult]
    final_metrics: Dict[str, float]
    memory_size_progression: List[int] = field(default_factory=list)
    timestamp: str = ""

    def __post_init__(self):
        if not self.timestamp:
            self.timestamp = datetime.now().isoformat()


class Evaluator:
    """
    Main evaluator for Evo-Memory benchmark.

    Handles both single-turn and multi-turn evaluations
    with streaming task presentation.
    """

    def __init__(
        self,
        agent: BaseAgent,
        dataset: BaseDataset,
        config: EvaluationConfig = None,
    ):
        """
        Initialize evaluator.

        Args:
            agent: Agent to evaluate
            dataset: Dataset to use
            config: Evaluation configuration
        """
        self.agent = agent
        self.dataset = dataset
        self.config = config or EvaluationConfig()

        # Select appropriate metric
        self.metric = self._select_metric()

        # Results storage
        self.results: List[TaskResult] = []
        self.stream_metrics = StreamMetrics()

    def _select_metric(self) -> BaseMetric:
        """Select metric based on dataset type."""
        dataset_name = self.dataset.name.lower()

        if "aime" in dataset_name:
            return AIMEAccuracy()
        elif "toolbench" in dataset_name:
            return ToolBenchAccuracy()
        elif isinstance(self.dataset, MultiTurnDataset):
            return SuccessRate()
        else:
            return AnswerAccuracy()

    def evaluate_stream(
        self,
        stream_id: str = "default",
        task_limit: Optional[int] = None,
    ) -> StreamResult:
        """
        Evaluate agent on a streaming task sequence.

        Args:
            stream_id: Identifier for this stream
            task_limit: Maximum number of tasks to evaluate

        Returns:
            StreamResult with all metrics
        """
        tasks = list(self.dataset)
        if task_limit:
            tasks = tasks[:task_limit]

        logger.info(f"Starting evaluation stream '{stream_id}' with {len(tasks)} tasks")

        task_results = []
        memory_progression = []

        for idx, task in enumerate(tasks):
            if self.config.verbose:
                logger.info(f"Task {idx + 1}/{len(tasks)}: {task.task_id}")

            # Evaluate single task
            result = self._evaluate_task(task)
            task_results.append(result)

            # Track memory size
            memory_size = len(self.agent.memory) if self.agent.memory else 0
            memory_progression.append(memory_size)

            # Log progress
            if self.config.verbose and (idx + 1) % 10 == 0:
                recent_acc = sum(r.correct for r in task_results[-10:]) / 10
                logger.info(f"Progress: {idx + 1}/{len(tasks)}, Recent accuracy: {recent_acc:.2%}")

        # Compute final metrics
        final_metrics = self._compute_final_metrics(task_results)

        return StreamResult(
            stream_id=stream_id,
            agent_name=self.agent.__class__.__name__,
            dataset_name=self.dataset.name,
            task_results=task_results,
            final_metrics=final_metrics,
            memory_size_progression=memory_progression,
        )

    def _evaluate_task(self, task: TaskInstance) -> TaskResult:
        """
        Evaluate agent on a single task.

        Args:
            task: Task instance to evaluate

        Returns:
            TaskResult with score and trajectory
        """
        start_time = time.time()

        if isinstance(self.dataset, MultiTurnDataset):
            result = self._evaluate_multi_turn(task)
        else:
            result = self._evaluate_single_turn(task)

        result.time_taken = time.time() - start_time
        return result

    def _evaluate_single_turn(self, task: TaskInstance) -> TaskResult:
        """Evaluate single-turn task."""
        # Agent processes task
        trajectory = []
        prediction = ""

        try:
            # Search for relevant memories
            retrieved = self.agent.search(task.input_text)
            trajectory.append({
                "step": "search",
                "retrieved_count": len(retrieved),
            })

            # Synthesize response
            prediction = self.agent.synthesize(task.input_text, retrieved)
            trajectory.append({
                "step": "synthesize",
                "prediction": prediction,
            })

            # Evolve memory with result
            is_correct = self.metric.compute(prediction, task.target) >= 0.5
            self.agent.evolve(task.input_text, prediction, correct=is_correct)
            trajectory.append({
                "step": "evolve",
                "correct": is_correct,
            })

        except Exception as e:
            logger.error(f"Error evaluating task {task.task_id}: {e}")
            prediction = ""
            trajectory.append({"step": "error", "message": str(e)})

        # Compute score
        score = self.metric.compute(prediction, task.target)
        correct = score >= 0.5

        return TaskResult(
            task_id=task.task_id,
            input_text=task.input_text,
            prediction=prediction,
            target=task.target,
            correct=correct,
            score=score,
            steps=1,
            trajectory=trajectory if self.config.save_trajectories else [],
            memory_used=len(self.agent.memory) if self.agent.memory else 0,
        )

    def _evaluate_multi_turn(self, task: TaskInstance) -> TaskResult:
        """Evaluate multi-turn task with environment."""
        # Get environment
        env = self.dataset.get_environment(task)
        env_info = self.dataset.get_environment_info(task)

        # Initialize
        observation = env.reset()
        trajectory = []
        total_reward = 0.0
        done = False
        step = 0
        final_info = {"success": False, "progress": 0.0}

        while not done and step < self.config.max_steps:
            step += 1

            # Search for relevant memories
            context = f"{task.input_text}\n{observation}"
            retrieved = self.agent.search(context)

            # Synthesize action
            action = self.agent.synthesize(
                context,
                retrieved,
                environment_info=env_info,
            )

            trajectory.append({
                "step": step,
                "observation": observation,
                "action": action,
            })

            # Execute action
            observation, reward, done, info = env.step(action)
            total_reward += reward
            final_info = info

            trajectory[-1]["reward"] = reward
            trajectory[-1]["next_observation"] = observation

        # Evolve memory based on trajectory
        success = final_info.get("success", False)
        self.agent.evolve(
            task.input_text,
            f"Steps: {step}, Success: {success}",
            correct=success,
            trajectory=trajectory,
        )

        # Compute score (success rate)
        score = 1.0 if success else final_info.get("progress", 0.0)

        return TaskResult(
            task_id=task.task_id,
            input_text=task.input_text,
            prediction="success" if success else "failure",
            target=task.target,
            correct=success,
            score=score,
            steps=step,
            trajectory=trajectory if self.config.save_trajectories else [],
            memory_used=len(self.agent.memory) if self.agent.memory else 0,
            metadata={
                "total_reward": total_reward,
                "progress": final_info.get("progress", 0.0),
            },
        )

    def _compute_final_metrics(self, results: List[TaskResult]) -> Dict[str, float]:
        """Compute final metrics from results."""
        if not results:
            return {}

        metrics = {}

        # Basic metrics
        metrics["accuracy"] = sum(r.correct for r in results) / len(results)
        metrics["avg_score"] = sum(r.score for r in results) / len(results)
        metrics["total_tasks"] = len(results)
        metrics["correct_tasks"] = sum(r.correct for r in results)

        # For multi-turn tasks
        if any(r.steps > 1 for r in results):
            # Success rate
            metrics["success_rate"] = metrics["accuracy"]

            # Progress rate (average progress for all tasks)
            progress_values = [r.metadata.get("progress", r.score) for r in results]
            metrics["progress_rate"] = sum(progress_values) / len(progress_values)

            # Step efficiency (average steps for successful tasks)
            successful = [r for r in results if r.correct]
            if successful:
                metrics["avg_steps_success"] = sum(r.steps for r in successful) / len(successful)
            metrics["avg_steps_all"] = sum(r.steps for r in results) / len(results)

        # Learning curve metrics
        mid = len(results) // 2
        if mid > 0:
            early = results[:mid]
            late = results[mid:]
            metrics["early_accuracy"] = sum(r.correct for r in early) / len(early)
            metrics["late_accuracy"] = sum(r.correct for r in late) / len(late)
            metrics["learning_improvement"] = metrics["late_accuracy"] - metrics["early_accuracy"]

        # Memory metrics
        metrics["final_memory_size"] = results[-1].memory_used if results else 0

        return metrics


class MultiStreamEvaluator:
    """
    Evaluator for multiple streams/runs.

    Used to compute statistics across multiple evaluation runs.
    """

    def __init__(
        self,
        agent_factory,  # Callable that creates new agent
        dataset_factory,  # Callable that creates new dataset
        config: EvaluationConfig = None,
        num_streams: int = 3,
    ):
        self.agent_factory = agent_factory
        self.dataset_factory = dataset_factory
        self.config = config or EvaluationConfig()
        self.num_streams = num_streams

    def evaluate(self, task_limit: Optional[int] = None) -> Dict[str, Any]:
        """
        Run multiple evaluation streams.

        Returns:
            Aggregated statistics across all streams
        """
        all_results = []

        for stream_idx in range(self.num_streams):
            logger.info(f"Starting stream {stream_idx + 1}/{self.num_streams}")

            # Create fresh agent and dataset
            agent = self.agent_factory()
            dataset = self.dataset_factory()

            # Run evaluation
            evaluator = Evaluator(agent, dataset, self.config)
            result = evaluator.evaluate_stream(
                stream_id=f"stream_{stream_idx}",
                task_limit=task_limit,
            )
            all_results.append(result)

        # Aggregate across streams
        return self._aggregate_results(all_results)

    def _aggregate_results(self, results: List[StreamResult]) -> Dict[str, Any]:
        """Aggregate results across multiple streams."""
        if not results:
            return {}

        # Collect metrics from all streams
        metric_values = {}
        for result in results:
            for metric_name, value in result.final_metrics.items():
                if metric_name not in metric_values:
                    metric_values[metric_name] = []
                metric_values[metric_name].append(value)

        # Compute mean and std for each metric
        aggregated = {}
        for metric_name, values in metric_values.items():
            import numpy as np
            aggregated[f"{metric_name}_mean"] = float(np.mean(values))
            aggregated[f"{metric_name}_std"] = float(np.std(values))

        aggregated["num_streams"] = len(results)

        return aggregated


def save_results(result: StreamResult, output_path: str):
    """Save evaluation results to file."""
    output = {
        "stream_id": result.stream_id,
        "agent_name": result.agent_name,
        "dataset_name": result.dataset_name,
        "timestamp": result.timestamp,
        "final_metrics": result.final_metrics,
        "memory_progression": result.memory_size_progression,
        "task_results": [
            {
                "task_id": r.task_id,
                "prediction": r.prediction,
                "target": r.target,
                "correct": r.correct,
                "score": r.score,
                "steps": r.steps,
                "time_taken": r.time_taken,
                "trajectory": r.trajectory,
            }
            for r in result.task_results
        ],
    }

    with open(output_path, 'w') as f:
        json.dump(output, f, indent=2)

    logger.info(f"Results saved to {output_path}")


def load_results(input_path: str) -> Dict[str, Any]:
    """Load evaluation results from file."""
    with open(input_path, 'r') as f:
        return json.load(f)
