"""MMLU-Pro dataset loader.

MMLU-Pro extends the original MMLU benchmark with stronger robustness
and challenge, testing multi-disciplinary reasoning across domains
such as engineering, philosophy, and economics.
"""

from typing import List, Dict, Any, Optional
import json
import re

from ..base import SingleTurnDataset, TaskInstance, DatasetSplit


class MMLUProDataset(SingleTurnDataset):
    """
    MMLU-Pro dataset for multi-disciplinary reasoning evaluation.

    Supports domain filtering for focused evaluation:
    - engineering
    - philosophy
    - economics
    - and other MMLU domains
    """

    def __init__(
        self,
        data_path: Optional[str] = None,
        split: DatasetSplit = DatasetSplit.TEST,
        domain: Optional[str] = None,
        **kwargs,
    ):
        """
        Initialize MMLU-Pro dataset.

        Args:
            data_path: Path to MMLU-Pro data
            split: Dataset split
            domain: Specific domain to load (e.g., 'engineering', 'philosophy')
        """
        super().__init__(data_path, split, **kwargs)
        self.domain_filter = domain

    @property
    def name(self) -> str:
        if self.domain_filter:
            return f"mmlu_pro_{self.domain_filter}"
        return "mmlu_pro"

    def _load_data(self) -> List[TaskInstance]:
        """Load MMLU-Pro data from HuggingFace or local path."""
        instances = []

        try:
            from datasets import load_dataset

            # Load from HuggingFace
            dataset = load_dataset("TIGER-Lab/MMLU-Pro", split=self.split.value)

            for idx, item in enumerate(dataset):
                domain = item.get("category", "general")

                # Apply domain filter
                if self.domain_filter and domain.lower() != self.domain_filter.lower():
                    continue

                # Build question with options
                question = item["question"]
                options = item.get("options", [])
                if options:
                    options_text = "\n".join([
                        f"{chr(65+i)}. {opt}" for i, opt in enumerate(options)
                    ])
                    question = f"{question}\n\nOptions:\n{options_text}"

                instances.append(TaskInstance(
                    task_id=f"mmlu_pro_{idx}",
                    input_text=question,
                    target=item.get("answer", ""),
                    metadata={
                        "options": options,
                        "answer_index": item.get("answer_index"),
                    },
                    domain=domain,
                    difficulty=self._estimate_difficulty(item),
                ))

        except Exception as e:
            print(f"Warning: Could not load MMLU-Pro from HuggingFace: {e}")
            # Load from local path if available
            if self.data_path:
                instances = self._load_from_local()

        return instances

    def _load_from_local(self) -> List[TaskInstance]:
        """Load from local JSON file."""
        instances = []

        try:
            with open(self.data_path) as f:
                data = json.load(f)

            for idx, item in enumerate(data):
                domain = item.get("category", "general")

                if self.domain_filter and domain.lower() != self.domain_filter.lower():
                    continue

                question = item["question"]
                options = item.get("options", [])
                if options:
                    options_text = "\n".join([
                        f"{chr(65+i)}. {opt}" for i, opt in enumerate(options)
                    ])
                    question = f"{question}\n\nOptions:\n{options_text}"

                instances.append(TaskInstance(
                    task_id=f"mmlu_pro_{idx}",
                    input_text=question,
                    target=item.get("answer", ""),
                    metadata={"options": options},
                    domain=domain,
                ))

        except FileNotFoundError:
            print(f"Warning: Local data file not found: {self.data_path}")

        return instances

    def _estimate_difficulty(self, item: Dict) -> str:
        """Estimate question difficulty based on heuristics."""
        question = item.get("question", "")
        # Simple heuristic based on question length and complexity
        if len(question) > 500:
            return "hard"
        elif len(question) > 200:
            return "medium"
        return "easy"

    def evaluate(self, prediction: str, target: str) -> Dict[str, Any]:
        """
        Evaluate MMLU-Pro prediction.

        Supports both letter answers (A, B, C, D) and full text answers.
        """
        prediction = prediction.strip().upper()
        target = target.strip().upper()

        # Extract letter if present
        letter_match = re.search(r"^([A-D])", prediction)
        if letter_match:
            prediction = letter_match.group(1)

        is_correct = prediction == target

        return {
            "correct": is_correct,
            "exact_match": is_correct,
            "prediction": prediction,
            "target": target,
        }
