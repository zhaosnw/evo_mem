"""Base dataset classes for Evo-Memory.

Evo-Memory restructures conventional static datasets into streaming task
sequences, enabling evaluation of how LLMs reuse and evolve memory over time.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional, Iterator, Callable
from enum import Enum
import random


class DatasetSplit(Enum):
    """Dataset splits."""
    TRAIN = "train"
    VAL = "validation"
    TEST = "test"


@dataclass
class TaskInstance:
    """
    A single task instance in the streaming evaluation.

    Attributes:
        task_id: Unique identifier for this task
        input_text: The input query/question (x_t)
        target: The expected output/answer (y_t)
        metadata: Additional task metadata
        difficulty: Optional difficulty level (easy/medium/hard)
        domain: Task domain/category
    """
    task_id: str
    input_text: str
    target: str
    metadata: Dict[str, Any] = field(default_factory=dict)
    difficulty: Optional[str] = None
    domain: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "task_id": self.task_id,
            "input_text": self.input_text,
            "target": self.target,
            "metadata": self.metadata,
            "difficulty": self.difficulty,
            "domain": self.domain,
        }


class BaseDataset(ABC):
    """
    Abstract base class for Evo-Memory datasets.

    Datasets are converted into streaming task sequences:
    τ = {(x_1, y_1), ..., (x_T, y_T)}
    """

    def __init__(
        self,
        data_path: Optional[str] = None,
        split: DatasetSplit = DatasetSplit.TEST,
        seed: int = 42,
        shuffle: bool = False,
        max_samples: Optional[int] = None,
    ):
        """
        Initialize dataset.

        Args:
            data_path: Path to dataset files
            split: Dataset split to use
            seed: Random seed for shuffling
            shuffle: Whether to shuffle the data
            max_samples: Maximum number of samples to use
        """
        self.data_path = data_path
        self.split = split
        self.seed = seed
        self.shuffle = shuffle
        self.max_samples = max_samples

        self._instances: List[TaskInstance] = []
        self._loaded = False

    @property
    @abstractmethod
    def name(self) -> str:
        """Dataset name."""
        pass

    @property
    @abstractmethod
    def task_type(self) -> str:
        """Task type: 'single_turn' or 'multi_turn'."""
        pass

    @abstractmethod
    def _load_data(self) -> List[TaskInstance]:
        """Load data from source. To be implemented by subclasses."""
        pass

    @abstractmethod
    def evaluate(self, prediction: str, target: str) -> Dict[str, Any]:
        """
        Evaluate a single prediction.

        Args:
            prediction: Model prediction
            target: Ground truth target

        Returns:
            Dictionary with evaluation metrics
        """
        pass

    def load(self) -> None:
        """Load the dataset."""
        if self._loaded:
            return

        self._instances = self._load_data()

        if self.shuffle:
            random.seed(self.seed)
            random.shuffle(self._instances)

        if self.max_samples:
            self._instances = self._instances[:self.max_samples]

        self._loaded = True

    def __len__(self) -> int:
        if not self._loaded:
            self.load()
        return len(self._instances)

    def __iter__(self) -> Iterator[TaskInstance]:
        """Iterate over task instances in streaming fashion."""
        if not self._loaded:
            self.load()
        return iter(self._instances)

    def __getitem__(self, idx: int) -> TaskInstance:
        if not self._loaded:
            self.load()
        return self._instances[idx]

    def get_stream(
        self,
        order: str = "default",
        difficulty_order: Optional[str] = None,
    ) -> Iterator[TaskInstance]:
        """
        Get task stream with specified ordering.

        Args:
            order: 'default', 'random', 'difficulty'
            difficulty_order: 'easy_to_hard' or 'hard_to_easy' (if order='difficulty')

        Returns:
            Iterator over task instances
        """
        if not self._loaded:
            self.load()

        instances = list(self._instances)

        if order == "random":
            random.shuffle(instances)
        elif order == "difficulty" and difficulty_order:
            difficulty_map = {"easy": 0, "medium": 1, "hard": 2}
            reverse = difficulty_order == "hard_to_easy"
            instances.sort(
                key=lambda x: difficulty_map.get(x.difficulty or "medium", 1),
                reverse=reverse,
            )

        return iter(instances)

    def get_statistics(self) -> Dict[str, Any]:
        """Get dataset statistics."""
        if not self._loaded:
            self.load()

        # Count by domain
        domains = {}
        difficulties = {}

        for inst in self._instances:
            domain = inst.domain or "unknown"
            domains[domain] = domains.get(domain, 0) + 1

            diff = inst.difficulty or "unknown"
            difficulties[diff] = difficulties.get(diff, 0) + 1

        return {
            "name": self.name,
            "task_type": self.task_type,
            "total_instances": len(self._instances),
            "domains": domains,
            "difficulties": difficulties,
        }


class SingleTurnDataset(BaseDataset):
    """Base class for single-turn datasets (QA, reasoning, etc.)."""

    @property
    def task_type(self) -> str:
        return "single_turn"


class MultiTurnDataset(BaseDataset):
    """Base class for multi-turn datasets (embodied, interactive, etc.)."""

    @property
    def task_type(self) -> str:
        return "multi_turn"

    @abstractmethod
    def get_environment(self, task_instance: TaskInstance):
        """
        Get environment interface for this task.

        Returns an object with:
        - reset() -> observation
        - step(action) -> (observation, reward, done, info)
        """
        pass

    @abstractmethod
    def get_environment_info(self, task_instance: TaskInstance) -> str:
        """Get environment description/instructions."""
        pass
