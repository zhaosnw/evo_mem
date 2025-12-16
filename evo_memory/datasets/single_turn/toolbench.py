"""ToolBench dataset loader.

ToolBench assesses a model's ability to identify and configure
external APIs, reflecting practical tool-use capabilities.
"""

from typing import List, Dict, Any, Optional
import json
import re

from ..base import SingleTurnDataset, TaskInstance, DatasetSplit


class ToolBenchDataset(SingleTurnDataset):
    """
    ToolBench dataset for tool-use and API grounding.

    Evaluates the model's ability to:
    - Select appropriate APIs
    - Configure API parameters correctly
    """

    def __init__(
        self,
        data_path: Optional[str] = None,
        split: DatasetSplit = DatasetSplit.TEST,
        category: Optional[str] = None,  # Filter by tool category
        **kwargs,
    ):
        """
        Initialize ToolBench dataset.

        Args:
            data_path: Path to ToolBench data
            split: Dataset split
            category: Tool category filter
        """
        super().__init__(data_path, split, **kwargs)
        self.category = category

    @property
    def name(self) -> str:
        if self.category:
            return f"toolbench_{self.category}"
        return "toolbench"

    def _load_data(self) -> List[TaskInstance]:
        """Load ToolBench data."""
        instances = []

        try:
            from datasets import load_dataset

            # Load from HuggingFace (using Gorilla/Berkeley format)
            dataset = load_dataset(
                "gorilla-llm/Berkeley-Function-Calling-Leaderboard",
                split="train",
            )

            for idx, item in enumerate(dataset):
                # Extract question and expected API call
                question = item.get("question", "")
                if not question:
                    continue

                # Get ground truth API
                ground_truth = item.get("ground_truth", item.get("answer", ""))

                # Get available functions
                functions = item.get("functions", [])

                # Build input with function descriptions
                functions_text = ""
                if functions:
                    func_descs = []
                    for func in functions[:5]:  # Limit functions shown
                        if isinstance(func, dict):
                            func_descs.append(
                                f"- {func.get('name', 'unknown')}: {func.get('description', '')[:100]}"
                            )
                    functions_text = "\n\nAvailable APIs:\n" + "\n".join(func_descs)

                full_input = f"{question}{functions_text}"

                instances.append(TaskInstance(
                    task_id=f"toolbench_{idx}",
                    input_text=full_input,
                    target=ground_truth if isinstance(ground_truth, str) else json.dumps(ground_truth),
                    metadata={
                        "functions": functions,
                        "category": item.get("category", "general"),
                    },
                    domain=item.get("category", "api"),
                    difficulty=self._estimate_difficulty(item),
                ))

        except Exception as e:
            print(f"Warning: Could not load ToolBench: {e}")
            if self.data_path:
                instances = self._load_from_local()
            else:
                instances = self._create_sample_tasks()

        return instances

    def _load_from_local(self) -> List[TaskInstance]:
        """Load from local JSON file."""
        instances = []

        try:
            with open(self.data_path) as f:
                data = json.load(f)

            for idx, item in enumerate(data):
                instances.append(TaskInstance(
                    task_id=f"toolbench_{idx}",
                    input_text=item["question"],
                    target=item.get("answer", ""),
                    metadata=item.get("metadata", {}),
                    domain="api",
                ))

        except FileNotFoundError:
            print(f"Warning: Local data file not found: {self.data_path}")

        return instances

    def _create_sample_tasks(self) -> List[TaskInstance]:
        """Create sample ToolBench tasks for testing."""
        tasks = [
            {
                "question": "What's the weather like in San Francisco today?",
                "functions": [
                    {"name": "get_weather", "description": "Get weather for a location"},
                    {"name": "search_web", "description": "Search the web"},
                ],
                "answer": '{"name": "get_weather", "arguments": {"location": "San Francisco"}}',
            },
            {
                "question": "Send an email to john@example.com with subject 'Meeting'",
                "functions": [
                    {"name": "send_email", "description": "Send an email"},
                    {"name": "create_calendar_event", "description": "Create calendar event"},
                ],
                "answer": '{"name": "send_email", "arguments": {"to": "john@example.com", "subject": "Meeting"}}',
            },
        ]

        instances = []
        for idx, task in enumerate(tasks):
            func_text = "\n".join([
                f"- {f['name']}: {f['description']}"
                for f in task["functions"]
            ])
            full_input = f"{task['question']}\n\nAvailable APIs:\n{func_text}"

            instances.append(TaskInstance(
                task_id=f"toolbench_{idx}",
                input_text=full_input,
                target=task["answer"],
                metadata={"functions": task["functions"]},
                domain="api",
            ))

        return instances

    def _estimate_difficulty(self, item: Dict) -> str:
        """Estimate task difficulty."""
        functions = item.get("functions", [])
        if len(functions) > 5:
            return "hard"
        elif len(functions) > 2:
            return "medium"
        return "easy"

    def evaluate(self, prediction: str, target: str) -> Dict[str, Any]:
        """
        Evaluate ToolBench prediction.

        Metrics:
        - API accuracy: Correct API selected
        - Full accuracy: API + parameters correct
        """
        # Parse prediction and target as JSON
        try:
            pred_obj = json.loads(prediction) if isinstance(prediction, str) else prediction
        except json.JSONDecodeError:
            # Try to extract API name from text
            api_match = re.search(r'"name":\s*"(\w+)"', prediction)
            pred_obj = {"name": api_match.group(1)} if api_match else {"name": ""}

        try:
            target_obj = json.loads(target) if isinstance(target, str) else target
        except json.JSONDecodeError:
            target_obj = {"name": target}

        # Check API name match
        pred_name = pred_obj.get("name", "") if isinstance(pred_obj, dict) else ""
        target_name = target_obj.get("name", "") if isinstance(target_obj, dict) else ""

        api_correct = pred_name.lower() == target_name.lower()

        # Check full match (including arguments)
        full_correct = pred_obj == target_obj if api_correct else False

        return {
            "correct": full_correct,
            "api_accuracy": api_correct,
            "exact_match": full_correct,
            "prediction": prediction,
            "target": target,
        }
