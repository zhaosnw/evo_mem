"""Evaluation metrics for Evo-Memory benchmark.

Implements metrics described in the paper:
- Answer Accuracy (single-turn)
- Success Rate (multi-turn)
- Progress Rate (multi-turn)
- Step Efficiency (multi-turn)
"""

from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass, field
from enum import Enum
import numpy as np
import re


class MetricType(Enum):
    """Types of evaluation metrics."""
    ACCURACY = "accuracy"
    SUCCESS_RATE = "success_rate"
    PROGRESS_RATE = "progress_rate"
    STEP_EFFICIENCY = "step_efficiency"


@dataclass
class EvaluationResult:
    """Result of a single task evaluation."""
    task_id: str
    correct: bool
    predicted: str
    target: str
    score: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class StreamEvaluationResult:
    """Result of evaluating a task stream."""
    stream_id: str
    task_results: List[EvaluationResult]
    metrics: Dict[str, float]

    @property
    def accuracy(self) -> float:
        """Compute accuracy over stream."""
        if not self.task_results:
            return 0.0
        return sum(r.correct for r in self.task_results) / len(self.task_results)

    @property
    def average_score(self) -> float:
        """Compute average score over stream."""
        if not self.task_results:
            return 0.0
        return sum(r.score for r in self.task_results) / len(self.task_results)


class BaseMetric:
    """Base class for evaluation metrics."""

    def __init__(self, name: str):
        self.name = name

    def compute(self, prediction: str, target: str, **kwargs) -> float:
        """Compute metric value."""
        raise NotImplementedError

    def aggregate(self, scores: List[float]) -> Dict[str, float]:
        """Aggregate scores across multiple instances."""
        if not scores:
            return {"mean": 0.0, "std": 0.0, "min": 0.0, "max": 0.0}
        return {
            "mean": float(np.mean(scores)),
            "std": float(np.std(scores)),
            "min": float(np.min(scores)),
            "max": float(np.max(scores)),
        }


class AnswerAccuracy(BaseMetric):
    """
    Answer accuracy for single-turn tasks.

    For multiple choice: exact match on option letter
    For open-ended: flexible matching with normalization
    """

    def __init__(self, strict: bool = False):
        super().__init__("answer_accuracy")
        self.strict = strict

    def compute(self, prediction: str, target: str, **kwargs) -> float:
        """
        Compute answer accuracy.

        Args:
            prediction: Model prediction
            target: Ground truth answer

        Returns:
            1.0 if correct, 0.0 otherwise
        """
        # Normalize both
        pred_normalized = self._normalize(prediction)
        target_normalized = self._normalize(target)

        # Exact match
        if pred_normalized == target_normalized:
            return 1.0

        # Try to extract option letter for MC questions
        pred_option = self._extract_option(prediction)
        target_option = self._extract_option(target)

        if pred_option and target_option:
            return 1.0 if pred_option == target_option else 0.0

        # For numerical answers
        pred_num = self._extract_number(prediction)
        target_num = self._extract_number(target)

        if pred_num is not None and target_num is not None:
            # Allow small tolerance for floating point
            if abs(pred_num - target_num) < 1e-6:
                return 1.0

        # Substring match for non-strict mode
        if not self.strict:
            if target_normalized in pred_normalized:
                return 1.0
            if pred_normalized in target_normalized:
                return 1.0

        return 0.0

    def _normalize(self, text: str) -> str:
        """Normalize text for comparison."""
        text = text.lower().strip()
        # Remove punctuation
        text = re.sub(r'[^\w\s]', '', text)
        # Collapse whitespace
        text = re.sub(r'\s+', ' ', text)
        return text

    def _extract_option(self, text: str) -> Optional[str]:
        """Extract option letter (A, B, C, D, etc.)."""
        text = text.strip().upper()

        # Direct single letter
        if len(text) == 1 and text.isalpha():
            return text

        # Patterns like "Answer: A" or "(A)" or "A."
        patterns = [
            r'(?:answer|option|choice)[:\s]*([A-Z])',
            r'\(([A-Z])\)',
            r'^([A-Z])[.\):]',
            r'([A-Z])\s*$',
        ]

        for pattern in patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                return match.group(1).upper()

        return None

    def _extract_number(self, text: str) -> Optional[float]:
        """Extract numerical value from text."""
        # Look for numbers (including negative and decimals)
        patterns = [
            r'[-+]?\d*\.?\d+',
        ]

        for pattern in patterns:
            matches = re.findall(pattern, text)
            if matches:
                try:
                    return float(matches[-1])  # Take last number
                except ValueError:
                    continue

        return None


class SuccessRate(BaseMetric):
    """
    Success rate for multi-turn tasks.

    Measures whether the agent successfully completed the task.
    """

    def __init__(self):
        super().__init__("success_rate")

    def compute(self, prediction: str, target: str, info: Dict = None, **kwargs) -> float:
        """
        Compute success rate.

        Args:
            prediction: Final state or prediction
            target: Expected target (usually "success")
            info: Additional info from environment

        Returns:
            1.0 if successful, 0.0 otherwise
        """
        if info and "success" in info:
            return 1.0 if info["success"] else 0.0

        # Fallback to string matching
        pred_lower = prediction.lower()
        return 1.0 if "success" in pred_lower or pred_lower == target.lower() else 0.0


class ProgressRate(BaseMetric):
    """
    Progress rate for multi-turn tasks.

    Measures partial progress toward task completion.
    """

    def __init__(self):
        super().__init__("progress_rate")

    def compute(self, prediction: str, target: str, info: Dict = None, **kwargs) -> float:
        """
        Compute progress rate.

        Args:
            prediction: Final state or prediction
            target: Expected target
            info: Additional info containing progress

        Returns:
            Progress value between 0.0 and 1.0
        """
        if info and "progress" in info:
            return float(info["progress"])

        # If success, return 1.0
        if info and info.get("success"):
            return 1.0

        return 0.0


class StepEfficiency(BaseMetric):
    """
    Step efficiency for multi-turn tasks.

    Measures how efficiently the agent completed the task
    in terms of number of steps.
    """

    def __init__(self, optimal_steps: int = 10):
        super().__init__("step_efficiency")
        self.optimal_steps = optimal_steps

    def compute(
        self,
        prediction: str,
        target: str,
        info: Dict = None,
        steps_taken: int = None,
        **kwargs
    ) -> float:
        """
        Compute step efficiency.

        Args:
            prediction: Final state
            target: Expected target
            info: Additional info
            steps_taken: Number of steps taken

        Returns:
            Efficiency score (higher is better)
        """
        if steps_taken is None:
            if info and "steps" in info:
                steps_taken = info["steps"]
            else:
                return 0.0

        # Check if successful
        success = False
        if info and "success" in info:
            success = info["success"]

        if not success:
            return 0.0

        # Efficiency = optimal_steps / actual_steps (capped at 1.0)
        if steps_taken == 0:
            return 1.0

        efficiency = min(self.optimal_steps / steps_taken, 1.0)
        return efficiency


class AIMEAccuracy(BaseMetric):
    """
    Accuracy metric specifically for AIME problems.

    AIME answers are integers from 0-999.
    """

    def __init__(self):
        super().__init__("aime_accuracy")

    def compute(self, prediction: str, target: str, **kwargs) -> float:
        """
        Compute AIME accuracy.

        Args:
            prediction: Model prediction
            target: Ground truth (integer 0-999)

        Returns:
            1.0 if correct, 0.0 otherwise
        """
        # Extract integer from prediction
        pred_int = self._extract_integer(prediction)
        target_int = self._extract_integer(target)

        if pred_int is None or target_int is None:
            return 0.0

        return 1.0 if pred_int == target_int else 0.0

    def _extract_integer(self, text: str) -> Optional[int]:
        """Extract integer from text."""
        # Look for boxed answer first (LaTeX style)
        boxed_match = re.search(r'\\boxed\{(\d+)\}', text)
        if boxed_match:
            return int(boxed_match.group(1))

        # Look for "answer is X" pattern
        answer_match = re.search(r'(?:answer|result)(?:\s+is)?[:\s]*(\d+)', text, re.IGNORECASE)
        if answer_match:
            return int(answer_match.group(1))

        # Look for final number
        numbers = re.findall(r'\b(\d+)\b', text)
        if numbers:
            val = int(numbers[-1])
            if 0 <= val <= 999:  # Valid AIME range
                return val

        return None


class ToolBenchAccuracy(BaseMetric):
    """
    Accuracy metric for ToolBench API tasks.

    Evaluates API call correctness.
    """

    def __init__(self):
        super().__init__("toolbench_accuracy")

    def compute(self, prediction: str, target: str, **kwargs) -> float:
        """
        Compute ToolBench accuracy.

        Args:
            prediction: Model's API call prediction
            target: Expected API call or format

        Returns:
            Score based on API correctness
        """
        pred_lower = prediction.lower()
        target_lower = target.lower()

        # Extract API name
        pred_api = self._extract_api_name(pred_lower)
        target_api = self._extract_api_name(target_lower)

        if pred_api and target_api:
            # API name match
            if pred_api == target_api:
                # Check parameters
                pred_params = self._extract_parameters(prediction)
                target_params = self._extract_parameters(target)

                if self._params_match(pred_params, target_params):
                    return 1.0
                return 0.5  # Partial credit for correct API

        return 0.0

    def _extract_api_name(self, text: str) -> Optional[str]:
        """Extract API name from text."""
        patterns = [
            r'api[:\s]*([a-z_]+)',
            r'call[:\s]*([a-z_]+)',
            r'function[:\s]*([a-z_]+)',
        ]

        for pattern in patterns:
            match = re.search(pattern, text)
            if match:
                return match.group(1)

        return None

    def _extract_parameters(self, text: str) -> Dict[str, str]:
        """Extract parameters from API call."""
        params = {}

        # Look for key=value or "key": "value" patterns
        patterns = [
            r'(\w+)\s*[=:]\s*["\']?([^,\s\'"]+)["\']?',
        ]

        for pattern in patterns:
            matches = re.findall(pattern, text)
            for key, value in matches:
                params[key.lower()] = value.lower()

        return params

    def _params_match(self, pred: Dict, target: Dict) -> bool:
        """Check if parameters match."""
        if not target:
            return True

        for key, value in target.items():
            if key not in pred or pred[key] != value:
                return False

        return True


class StreamMetrics:
    """
    Compute metrics across a task stream.

    Tracks evolution of performance over time.
    """

    def __init__(self, metrics: List[BaseMetric] = None):
        self.metrics = metrics or [AnswerAccuracy()]
        self.results: List[EvaluationResult] = []

    def add_result(self, result: EvaluationResult):
        """Add evaluation result."""
        self.results.append(result)

    def compute_all(self) -> Dict[str, float]:
        """Compute all metrics."""
        if not self.results:
            return {}

        output = {}

        # Accuracy/Success rate
        correct_count = sum(r.correct for r in self.results)
        output["accuracy"] = correct_count / len(self.results)

        # Average score
        output["avg_score"] = sum(r.score for r in self.results) / len(self.results)

        # Score progression (early vs late)
        mid = len(self.results) // 2
        if mid > 0:
            early_scores = [r.score for r in self.results[:mid]]
            late_scores = [r.score for r in self.results[mid:]]
            output["early_avg"] = sum(early_scores) / len(early_scores)
            output["late_avg"] = sum(late_scores) / len(late_scores)
            output["improvement"] = output["late_avg"] - output["early_avg"]

        return output

    def get_learning_curve(self, window_size: int = 5) -> List[float]:
        """
        Get learning curve with moving average.

        Args:
            window_size: Window size for moving average

        Returns:
            List of averaged scores
        """
        if not self.results:
            return []

        scores = [r.score for r in self.results]
        curve = []

        for i in range(len(scores)):
            start = max(0, i - window_size + 1)
            window = scores[start:i + 1]
            curve.append(sum(window) / len(window))

        return curve


def compute_metrics(
    predictions: List[str],
    targets: List[str],
    metric_type: MetricType = MetricType.ACCURACY,
    infos: List[Dict] = None,
    **kwargs
) -> Dict[str, float]:
    """
    Compute metrics for a list of predictions.

    Args:
        predictions: List of model predictions
        targets: List of ground truth values
        metric_type: Type of metric to compute
        infos: Additional info for each prediction

    Returns:
        Dictionary of metric values
    """
    if len(predictions) != len(targets):
        raise ValueError("Predictions and targets must have same length")

    if infos is None:
        infos = [{}] * len(predictions)

    # Select metric
    if metric_type == MetricType.ACCURACY:
        metric = AnswerAccuracy()
    elif metric_type == MetricType.SUCCESS_RATE:
        metric = SuccessRate()
    elif metric_type == MetricType.PROGRESS_RATE:
        metric = ProgressRate()
    elif metric_type == MetricType.STEP_EFFICIENCY:
        metric = StepEfficiency()
    else:
        raise ValueError(f"Unknown metric type: {metric_type}")

    # Compute scores
    scores = []
    for pred, target, info in zip(predictions, targets, infos):
        score = metric.compute(pred, target, info=info, **kwargs)
        scores.append(score)

    # Aggregate
    aggregated = metric.aggregate(scores)
    aggregated["total"] = len(scores)
    aggregated["correct"] = sum(1 for s in scores if s >= 0.5)

    return aggregated
