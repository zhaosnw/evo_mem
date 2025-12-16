"""GPQA Diamond dataset loader.

GPQA-Diamond is a graduate-level benchmark featuring expert-written,
"Google-proof" questions in physics and related sciences.
"""

from typing import List, Dict, Any, Optional
import json
import re

from ..base import SingleTurnDataset, TaskInstance, DatasetSplit


class GPQADataset(SingleTurnDataset):
    """
    GPQA-Diamond dataset for graduate-level reasoning.

    Features expert-written questions that require rigorous
    multi-step reasoning in science domains.
    """

    def __init__(
        self,
        data_path: Optional[str] = None,
        split: DatasetSplit = DatasetSplit.TEST,
        subset: str = "diamond",  # diamond, extended, main
        **kwargs,
    ):
        """
        Initialize GPQA dataset.

        Args:
            data_path: Path to GPQA data
            split: Dataset split
            subset: GPQA subset ('diamond', 'extended', 'main')
        """
        super().__init__(data_path, split, **kwargs)
        self.subset = subset

    @property
    def name(self) -> str:
        return f"gpqa_{self.subset}"

    def _load_data(self) -> List[TaskInstance]:
        """Load GPQA data."""
        instances = []

        try:
            from datasets import load_dataset

            # GPQA on HuggingFace
            dataset = load_dataset(
                "Idavidrein/gpqa",
                self.subset,
                split=self.split.value if self.split != DatasetSplit.TEST else "train",
            )

            for idx, item in enumerate(dataset):
                # Build question with multiple choice options
                question = item["Question"]

                # Extract options
                options = []
                for key in ["Correct Answer", "Incorrect Answer 1",
                           "Incorrect Answer 2", "Incorrect Answer 3"]:
                    if key in item and item[key]:
                        options.append(item[key])

                # Shuffle options but track correct answer
                import random
                random.seed(idx)
                correct_answer = item.get("Correct Answer", "")
                random.shuffle(options)
                correct_idx = options.index(correct_answer) if correct_answer in options else 0

                options_text = "\n".join([
                    f"{chr(65+i)}. {opt}" for i, opt in enumerate(options)
                ])
                full_question = f"{question}\n\nOptions:\n{options_text}"

                instances.append(TaskInstance(
                    task_id=f"gpqa_{self.subset}_{idx}",
                    input_text=full_question,
                    target=chr(65 + correct_idx),  # A, B, C, or D
                    metadata={
                        "options": options,
                        "correct_answer": correct_answer,
                        "subdomain": item.get("High-level domain", "science"),
                    },
                    domain=item.get("High-level domain", "science"),
                    difficulty="hard",  # GPQA is designed to be challenging
                ))

        except Exception as e:
            print(f"Warning: Could not load GPQA: {e}")
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
                instances.append(TaskInstance(
                    task_id=f"gpqa_{self.subset}_{idx}",
                    input_text=item["question"],
                    target=item.get("answer", ""),
                    metadata=item.get("metadata", {}),
                    domain=item.get("domain", "science"),
                    difficulty="hard",
                ))

        except FileNotFoundError:
            print(f"Warning: Local data file not found: {self.data_path}")

        return instances

    def evaluate(self, prediction: str, target: str) -> Dict[str, Any]:
        """Evaluate GPQA prediction."""
        prediction = prediction.strip().upper()
        target = target.strip().upper()

        # Extract letter answer
        letter_match = re.search(r"([A-D])", prediction)
        if letter_match:
            prediction = letter_match.group(1)

        is_correct = prediction == target

        return {
            "correct": is_correct,
            "exact_match": is_correct,
            "prediction": prediction,
            "target": target,
        }
