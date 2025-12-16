"""AlfWorld dataset loader.

AlfWorld aligns text and embodied environments for interactive learning,
featuring household instruction following tasks.
"""

from typing import List, Dict, Any, Optional, Tuple
import json
import re
from dataclasses import dataclass

from ..base import MultiTurnDataset, TaskInstance, DatasetSplit


@dataclass
class AlfWorldTask:
    """AlfWorld task definition."""
    task_type: str  # pick, clean, heat, cool, examine, puttwo
    object_type: str
    receptacle_type: Optional[str] = None
    second_object: Optional[str] = None


class AlfWorldEnvironment:
    """
    Simulated AlfWorld environment interface.

    In a full implementation, this would connect to the actual
    AlfWorld TextWorld environment.
    """

    def __init__(self, task: TaskInstance):
        self.task = task
        self.task_def = self._parse_task(task.input_text)
        self.state = "initial"
        self.inventory = []
        self.current_location = "kitchen"
        self.step_count = 0
        self.max_steps = 50

        # Simulated world state
        self.objects = self._initialize_objects()
        self.valid_actions = self._get_valid_actions()

    def _parse_task(self, goal: str) -> AlfWorldTask:
        """Parse task goal into structured form."""
        goal_lower = goal.lower()

        if "put" in goal_lower and "in/on" in goal_lower:
            return AlfWorldTask(task_type="put", object_type="object", receptacle_type="receptacle")
        elif "clean" in goal_lower:
            return AlfWorldTask(task_type="clean", object_type="object")
        elif "heat" in goal_lower:
            return AlfWorldTask(task_type="heat", object_type="object")
        elif "cool" in goal_lower:
            return AlfWorldTask(task_type="cool", object_type="object")
        elif "examine" in goal_lower:
            return AlfWorldTask(task_type="examine", object_type="object")
        else:
            return AlfWorldTask(task_type="pick", object_type="object")

    def _initialize_objects(self) -> Dict[str, Dict]:
        """Initialize simulated objects."""
        return {
            "apple": {"location": "countertop", "state": "normal"},
            "mug": {"location": "cabinet", "state": "normal"},
            "plate": {"location": "sink", "state": "dirty"},
            "knife": {"location": "drawer", "state": "normal"},
        }

    def _get_valid_actions(self) -> List[str]:
        """Get currently valid actions."""
        actions = [
            "go to kitchen",
            "go to living room",
            "go to bedroom",
            "look around",
            "check inventory",
            "examine",
        ]

        for obj in self.objects:
            actions.extend([
                f"pick up {obj}",
                f"put {obj} in/on sink",
                f"use microwave",
                f"use fridge",
            ])

        return actions

    def reset(self) -> str:
        """Reset environment and return initial observation."""
        self.state = "initial"
        self.inventory = []
        self.step_count = 0
        self.current_location = "kitchen"

        return f"You are in the {self.current_location}. You see a countertop, sink, microwave, and fridge. Your task is: {self.task.input_text}"

    def step(self, action: str) -> Tuple[str, float, bool, Dict]:
        """
        Execute action in environment.

        Returns:
            observation: Text observation
            reward: Step reward
            done: Whether episode is done
            info: Additional info including success and progress
        """
        self.step_count += 1
        action_lower = action.lower().strip()

        # Check for max steps
        if self.step_count >= self.max_steps:
            return "Maximum steps reached.", 0.0, True, {"success": False, "progress": 0.5}

        # Process action
        observation = "Nothing happens."
        reward = 0.0
        done = False
        progress = min(self.step_count / 10, 0.9)

        if action_lower.startswith("go to"):
            location = action_lower.replace("go to", "").strip()
            self.current_location = location
            observation = f"You moved to the {location}."

        elif action_lower == "look around" or action_lower == "look":
            observation = f"You are in the {self.current_location}. You see various objects around."

        elif action_lower.startswith("pick up") or action_lower.startswith("take"):
            obj = action_lower.replace("pick up", "").replace("take", "").strip()
            if obj in self.objects:
                self.inventory.append(obj)
                observation = f"You picked up the {obj}."
                reward = 0.1
            else:
                observation = f"You don't see a {obj} here."

        elif action_lower.startswith("put"):
            if self.inventory:
                obj = self.inventory[-1]
                observation = f"You put the {obj} down."
                self.inventory.pop()
                # Check if task completed
                if self.task_def.task_type == "put":
                    done = True
                    reward = 1.0
            else:
                observation = "You're not holding anything."

        elif "use" in action_lower:
            if "microwave" in action_lower and self.task_def.task_type == "heat":
                if self.inventory:
                    observation = f"You heated the {self.inventory[-1]} in the microwave."
                    done = True
                    reward = 1.0
            elif "fridge" in action_lower and self.task_def.task_type == "cool":
                if self.inventory:
                    observation = f"You cooled the {self.inventory[-1]} in the fridge."
                    done = True
                    reward = 1.0
            else:
                observation = "You use the appliance but nothing special happens."

        elif action_lower == "check inventory" or action_lower == "inventory":
            if self.inventory:
                observation = f"You are carrying: {', '.join(self.inventory)}"
            else:
                observation = "Your inventory is empty."

        elif action_lower == "check valid actions":
            observation = "Valid actions: " + ", ".join(self.valid_actions[:10])

        info = {
            "success": done and reward > 0,
            "progress": progress if not done else (1.0 if reward > 0 else 0.5),
        }

        return observation, reward, done, info


class AlfWorldDataset(MultiTurnDataset):
    """
    AlfWorld dataset for household instruction following.

    Task types include:
    - pick_and_place: Pick up object and place somewhere
    - clean: Clean objects (wash in sink)
    - heat: Heat objects (use microwave)
    - cool: Cool objects (use fridge)
    - examine: Examine objects with lamp
    """

    def __init__(
        self,
        data_path: Optional[str] = None,
        split: DatasetSplit = DatasetSplit.TEST,
        task_types: Optional[List[str]] = None,
        **kwargs,
    ):
        """
        Initialize AlfWorld dataset.

        Args:
            data_path: Path to AlfWorld data
            split: Dataset split
            task_types: Filter by specific task types
        """
        super().__init__(data_path, split, **kwargs)
        self.task_types = task_types

    @property
    def name(self) -> str:
        return "alfworld"

    def _load_data(self) -> List[TaskInstance]:
        """Load AlfWorld tasks."""
        instances = []

        # Sample AlfWorld tasks
        tasks = [
            {"goal": "Put a hot apple in the fridge.", "type": "cool"},
            {"goal": "Put a clean mug in the cabinet.", "type": "clean"},
            {"goal": "Heat the plate and put it on the countertop.", "type": "heat"},
            {"goal": "Pick up the knife from the drawer.", "type": "pick"},
            {"goal": "Put the cooled tomato in the microwave.", "type": "put"},
            {"goal": "Clean the pan and put it on the stove.", "type": "clean"},
            {"goal": "Put a hot mug on the table.", "type": "heat"},
            {"goal": "Cool the apple and put it in the bowl.", "type": "cool"},
            {"goal": "Pick up the book from the shelf.", "type": "pick"},
            {"goal": "Put the cleaned plate in the cabinet.", "type": "clean"},
        ]

        # Load from file if available
        if self.data_path:
            try:
                with open(self.data_path) as f:
                    tasks = json.load(f)
            except FileNotFoundError:
                pass

        for idx, task in enumerate(tasks):
            task_type = task.get("type", "pick")

            if self.task_types and task_type not in self.task_types:
                continue

            instances.append(TaskInstance(
                task_id=f"alfworld_{idx}",
                input_text=task["goal"],
                target="success",  # Target is task completion
                metadata={
                    "task_type": task_type,
                    "game_file": task.get("game_file"),
                },
                domain="household",
                difficulty=self._estimate_difficulty(task),
            ))

        return instances

    def _estimate_difficulty(self, task: Dict) -> str:
        """Estimate task difficulty."""
        task_type = task.get("type", "pick")
        if task_type in ["pick", "examine"]:
            return "easy"
        elif task_type in ["clean", "heat", "cool"]:
            return "medium"
        return "hard"

    def get_environment(self, task_instance: TaskInstance) -> AlfWorldEnvironment:
        """Get AlfWorld environment for task."""
        return AlfWorldEnvironment(task_instance)

    def get_environment_info(self, task_instance: TaskInstance) -> str:
        """Get environment instructions."""
        return """You are in a household environment. You can perform these actions:
- go to [location]: Move to a location (kitchen, living room, bedroom, etc.)
- look around: See what's in the current location
- pick up [object]: Pick up an object
- put [object] in/on [receptacle]: Place object somewhere
- use [appliance]: Use microwave, fridge, sink, etc.
- check inventory: See what you're carrying
- check valid actions: List available actions"""

    def evaluate(self, prediction: str, target: str) -> Dict[str, Any]:
        """Evaluate based on task completion."""
        # For multi-turn, evaluation happens through environment
        return {
            "success": prediction.lower() == "success",
            "progress": 1.0 if prediction.lower() == "success" else 0.0,
        }
