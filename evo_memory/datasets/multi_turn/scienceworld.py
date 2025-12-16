"""ScienceWorld dataset loader.

ScienceWorld features open-ended scientific experimentation tasks,
testing reasoning about physical and chemical processes.
"""

from typing import List, Dict, Any, Optional, Tuple
import json
from dataclasses import dataclass

from ..base import MultiTurnDataset, TaskInstance, DatasetSplit


class ScienceWorldEnvironment:
    """
    Simulated ScienceWorld environment.

    Implements simplified science experiment scenarios.
    """

    def __init__(self, task: TaskInstance):
        self.task = task
        self.task_type = task.metadata.get("task_type", "general")
        self.step_count = 0
        self.max_steps = 50

        # Environment state
        self.inventory = []
        self.current_location = "lab"
        self.objects = self._initialize_objects()
        self.experiment_state = {}

    def _initialize_objects(self) -> Dict[str, Dict]:
        """Initialize objects based on task."""
        base_objects = {
            "thermometer": {"location": "lab", "type": "tool"},
            "beaker": {"location": "lab", "type": "container", "contains": None},
            "bunsen_burner": {"location": "lab", "type": "heat_source", "state": "off"},
            "ice": {"location": "freezer", "type": "substance", "state": "solid", "temp": 0},
            "water": {"location": "sink", "type": "substance", "state": "liquid", "temp": 20},
            "plant": {"location": "greenhouse", "type": "living", "state": "healthy"},
            "soil": {"location": "greenhouse", "type": "substance"},
            "seed": {"location": "storage", "type": "living", "state": "dormant"},
        }
        return base_objects

    def reset(self) -> str:
        """Reset environment."""
        self.step_count = 0
        self.inventory = []
        self.current_location = "lab"
        self.objects = self._initialize_objects()
        self.experiment_state = {}

        return self._get_observation()

    def _get_observation(self) -> str:
        """Get current observation."""
        parts = [f"Task: {self.task.input_text}"]
        parts.append(f"Location: {self.current_location}")

        # Objects in current location
        visible = [name for name, obj in self.objects.items()
                  if obj.get("location") == self.current_location]
        if visible:
            parts.append(f"You see: {', '.join(visible)}")

        if self.inventory:
            parts.append(f"Inventory: {', '.join(self.inventory)}")

        if self.experiment_state:
            parts.append(f"Experiment: {self.experiment_state}")

        return " | ".join(parts)

    def step(self, action: str) -> Tuple[str, float, bool, Dict]:
        """Execute action."""
        self.step_count += 1
        action_lower = action.lower().strip()

        if self.step_count >= self.max_steps:
            return "Max steps reached.", 0.0, True, {"success": False, "progress": 0.3}

        reward = 0.0
        done = False
        obs = "Nothing happens."

        # Go to location
        if action_lower.startswith("go to"):
            location = action_lower.replace("go to", "").strip()
            valid_locations = ["lab", "greenhouse", "storage", "sink", "freezer"]
            if location in valid_locations:
                self.current_location = location
                obs = f"You moved to the {location}."
            else:
                obs = f"Unknown location: {location}"

        # Pick up
        elif action_lower.startswith("pick up") or action_lower.startswith("take"):
            obj_name = action_lower.replace("pick up", "").replace("take", "").strip()
            if obj_name in self.objects:
                obj = self.objects[obj_name]
                if obj.get("location") == self.current_location:
                    self.inventory.append(obj_name)
                    obj["location"] = "inventory"
                    obs = f"You picked up the {obj_name}."
                    reward = 0.05
                else:
                    obs = f"The {obj_name} is not here."
            else:
                obs = f"No {obj_name} found."

        # Put/place
        elif action_lower.startswith("put") or action_lower.startswith("place"):
            parts = action_lower.replace("put", "").replace("place", "").strip()
            # Try to parse "X in/on Y"
            if " in " in parts or " on " in parts:
                for sep in [" in ", " on "]:
                    if sep in parts:
                        obj_name, container = parts.split(sep)
                        obj_name = obj_name.strip()
                        container = container.strip()
                        break

                if obj_name in self.inventory:
                    self.inventory.remove(obj_name)
                    self.objects[obj_name]["location"] = container
                    obs = f"You put the {obj_name} in/on the {container}."
                    reward = 0.1

        # Heat/cool
        elif action_lower.startswith("heat"):
            obj_name = action_lower.replace("heat", "").strip()
            if obj_name in self.objects:
                obj = self.objects[obj_name]
                obj["temp"] = obj.get("temp", 20) + 50
                if obj.get("state") == "solid" and obj["temp"] > 0:
                    obj["state"] = "liquid"
                    obs = f"The {obj_name} melted!"
                elif obj.get("state") == "liquid" and obj["temp"] > 100:
                    obj["state"] = "gas"
                    obs = f"The {obj_name} evaporated!"
                else:
                    obs = f"You heated the {obj_name}. Temperature: {obj['temp']}°C"
                reward = 0.2
                self.experiment_state["heated"] = obj_name

        elif action_lower.startswith("cool"):
            obj_name = action_lower.replace("cool", "").strip()
            if obj_name in self.objects:
                obj = self.objects[obj_name]
                obj["temp"] = obj.get("temp", 20) - 30
                if obj.get("state") == "liquid" and obj["temp"] < 0:
                    obj["state"] = "solid"
                    obs = f"The {obj_name} froze!"
                else:
                    obs = f"You cooled the {obj_name}. Temperature: {obj['temp']}°C"
                reward = 0.2
                self.experiment_state["cooled"] = obj_name

        # Measure
        elif action_lower.startswith("measure") or action_lower.startswith("use thermometer"):
            obj_name = action_lower.replace("measure", "").replace("use thermometer on", "").strip()
            if obj_name in self.objects:
                temp = self.objects[obj_name].get("temp", 20)
                obs = f"The {obj_name} is at {temp}°C."
                self.experiment_state["measured"] = {"object": obj_name, "temp": temp}
                reward = 0.1

        # Mix
        elif action_lower.startswith("mix"):
            if len(self.inventory) >= 2:
                obs = f"You mixed {' and '.join(self.inventory)}."
                self.experiment_state["mixed"] = list(self.inventory)
                reward = 0.2

        # Look/examine
        elif action_lower in ["look", "examine", "look around"]:
            obs = self._get_observation()

        # Check for task completion
        done, success = self._check_completion()
        if done:
            reward = 1.0 if success else 0.0
            obs += " Task completed!" if success else " Task failed."

        progress = self._calculate_progress()

        return obs, reward, done, {"success": success if done else False, "progress": progress}

    def _check_completion(self) -> Tuple[bool, bool]:
        """Check if task is completed."""
        task_type = self.task_type
        goal = self.task.input_text.lower()

        if "melt" in goal:
            for obj in self.objects.values():
                if obj.get("state") == "liquid" and "melted" not in str(obj):
                    if self.experiment_state.get("heated"):
                        return True, True

        if "freeze" in goal or "solidify" in goal:
            for obj in self.objects.values():
                if obj.get("state") == "solid" and obj.get("temp", 20) < 0:
                    return True, True

        if "measure" in goal and "temperature" in goal:
            if self.experiment_state.get("measured"):
                return True, True

        if "boil" in goal or "evaporate" in goal:
            for obj in self.objects.values():
                if obj.get("state") == "gas":
                    return True, True

        return False, False

    def _calculate_progress(self) -> float:
        """Calculate task progress."""
        progress = 0.0

        if self.experiment_state:
            progress += 0.3

        if self.inventory:
            progress += 0.1 * len(self.inventory)

        if any(obj.get("temp", 20) != 20 for obj in self.objects.values()):
            progress += 0.2

        return min(progress, 0.9)


class ScienceWorldDataset(MultiTurnDataset):
    """
    ScienceWorld dataset for scientific experimentation.

    Tasks involve physical, chemical, and biological processes.
    """

    def __init__(
        self,
        data_path: Optional[str] = None,
        split: DatasetSplit = DatasetSplit.TEST,
        task_types: Optional[List[str]] = None,
        **kwargs,
    ):
        """
        Initialize ScienceWorld dataset.

        Args:
            data_path: Path to ScienceWorld data
            split: Dataset split
            task_types: Filter by task types
        """
        super().__init__(data_path, split, **kwargs)
        self.task_types = task_types

    @property
    def name(self) -> str:
        return "scienceworld"

    def _load_data(self) -> List[TaskInstance]:
        """Load ScienceWorld tasks."""
        tasks = [
            {"goal": "Melt the ice and measure its temperature.", "type": "phase_change"},
            {"goal": "Freeze the water.", "type": "phase_change"},
            {"goal": "Boil water to create steam.", "type": "phase_change"},
            {"goal": "Measure the temperature of the water.", "type": "measurement"},
            {"goal": "Heat the beaker with water.", "type": "heating"},
            {"goal": "Cool the water below freezing.", "type": "cooling"},
            {"goal": "Grow a plant from a seed.", "type": "biology"},
            {"goal": "Mix two substances together.", "type": "chemistry"},
            {"goal": "Observe the state change of ice.", "type": "observation"},
            {"goal": "Conduct a temperature experiment.", "type": "experiment"},
        ]

        instances = []
        for idx, task in enumerate(tasks):
            task_type = task.get("type", "general")

            if self.task_types and task_type not in self.task_types:
                continue

            instances.append(TaskInstance(
                task_id=f"scienceworld_{idx}",
                input_text=task["goal"],
                target="success",
                metadata={"task_type": task_type},
                domain="science",
                difficulty=self._estimate_difficulty(task_type),
            ))

        return instances

    def _estimate_difficulty(self, task_type: str) -> str:
        """Estimate task difficulty."""
        easy = ["measurement", "observation"]
        medium = ["heating", "cooling", "phase_change"]
        hard = ["chemistry", "biology", "experiment"]

        if task_type in easy:
            return "easy"
        elif task_type in medium:
            return "medium"
        return "hard"

    def get_environment(self, task_instance: TaskInstance) -> ScienceWorldEnvironment:
        """Get ScienceWorld environment for task."""
        return ScienceWorldEnvironment(task_instance)

    def get_environment_info(self, task_instance: TaskInstance) -> str:
        """Get environment instructions."""
        return """You are in a science lab. Available actions:
- go to [location]: Move to lab, greenhouse, storage, sink, freezer
- pick up [object]: Pick up an object
- put [object] in/on [container]: Place object somewhere
- heat [object]: Apply heat to an object
- cool [object]: Cool an object
- measure [object]: Use thermometer to measure temperature
- mix: Mix items in inventory
- look: Observe surroundings

Objects may change state (solid/liquid/gas) based on temperature."""

    def evaluate(self, prediction: str, target: str) -> Dict[str, Any]:
        """Evaluate based on task completion."""
        return {
            "success": prediction.lower() == "success",
            "progress": 1.0 if prediction.lower() == "success" else 0.0,
        }
