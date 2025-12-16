"""BabyAI dataset loader.

BabyAI is a platform to study the sample efficiency of grounded
language learning, featuring navigation and compositional reasoning.
"""

from typing import List, Dict, Any, Optional, Tuple
import json
from dataclasses import dataclass

from ..base import MultiTurnDataset, TaskInstance, DatasetSplit


class BabyAIEnvironment:
    """
    Simulated BabyAI environment interface.

    Features a grid world with navigation tasks.
    """

    def __init__(self, task: TaskInstance):
        self.task = task
        self.goal = task.input_text
        self.step_count = 0
        self.max_steps = 64

        # Grid state (simplified)
        self.agent_pos = (3, 3)  # Center of 7x7 grid
        self.agent_dir = 0  # 0=right, 1=down, 2=left, 3=up
        self.grid_size = 7
        self.objects = self._initialize_objects()
        self.carrying = None

    def _initialize_objects(self) -> Dict[str, Dict]:
        """Initialize grid objects based on task."""
        objects = {
            "red_ball": {"pos": (1, 1), "type": "ball", "color": "red"},
            "blue_key": {"pos": (5, 2), "type": "key", "color": "blue"},
            "green_box": {"pos": (2, 5), "type": "box", "color": "green"},
            "yellow_door": {"pos": (6, 3), "type": "door", "color": "yellow", "open": False},
        }
        return objects

    def reset(self) -> str:
        """Reset environment."""
        self.step_count = 0
        self.agent_pos = (3, 3)
        self.agent_dir = 0
        self.carrying = None
        self.objects = self._initialize_objects()

        return self._get_observation()

    def _get_observation(self) -> str:
        """Get current observation."""
        # Find nearby objects
        nearby = []
        for name, obj in self.objects.items():
            dist = abs(obj["pos"][0] - self.agent_pos[0]) + abs(obj["pos"][1] - self.agent_pos[1])
            if dist <= 2:
                direction = self._get_direction(self.agent_pos, obj["pos"])
                nearby.append(f"{obj['color']} {obj['type']} to the {direction}")

        obs_parts = [f"Goal: {self.goal}"]
        obs_parts.append(f"Position: {self.agent_pos}")
        if nearby:
            obs_parts.append(f"You see: {', '.join(nearby)}")
        else:
            obs_parts.append("You see nothing nearby.")
        if self.carrying:
            obs_parts.append(f"Carrying: {self.carrying}")

        return " | ".join(obs_parts)

    def _get_direction(self, from_pos: Tuple[int, int], to_pos: Tuple[int, int]) -> str:
        """Get relative direction."""
        dx = to_pos[0] - from_pos[0]
        dy = to_pos[1] - from_pos[1]

        if abs(dx) > abs(dy):
            return "right" if dx > 0 else "left"
        else:
            return "down" if dy > 0 else "up"

    def step(self, action: str) -> Tuple[str, float, bool, Dict]:
        """Execute action."""
        self.step_count += 1
        action_lower = action.lower().strip()

        reward = 0.0
        done = False

        if self.step_count >= self.max_steps:
            return "Max steps reached.", 0.0, True, {"success": False, "progress": 0.3}

        # Parse action
        if action_lower in ["turn left", "left"]:
            self.agent_dir = (self.agent_dir - 1) % 4
            obs = "You turned left."

        elif action_lower in ["turn right", "right"]:
            self.agent_dir = (self.agent_dir + 1) % 4
            obs = "You turned right."

        elif action_lower in ["go forward", "forward", "move"]:
            dx, dy = [(1, 0), (0, 1), (-1, 0), (0, -1)][self.agent_dir]
            new_pos = (self.agent_pos[0] + dx, self.agent_pos[1] + dy)

            if 0 <= new_pos[0] < self.grid_size and 0 <= new_pos[1] < self.grid_size:
                self.agent_pos = new_pos
                obs = f"You moved to {new_pos}."
            else:
                obs = "You hit a wall."

        elif action_lower.startswith("pick up") or action_lower == "pickup":
            # Find pickable object at current position
            for name, obj in self.objects.items():
                if obj["pos"] == self.agent_pos and obj["type"] in ["ball", "key", "box"]:
                    self.carrying = name
                    obs = f"You picked up the {obj['color']} {obj['type']}."
                    reward = 0.2
                    break
            else:
                obs = "Nothing to pick up here."

        elif action_lower == "drop":
            if self.carrying:
                obj_name = self.carrying
                self.objects[obj_name]["pos"] = self.agent_pos
                obs = f"You dropped the {obj_name}."
                self.carrying = None
            else:
                obs = "You're not carrying anything."

        elif action_lower == "toggle":
            # Toggle door
            for name, obj in self.objects.items():
                if obj["type"] == "door":
                    dist = abs(obj["pos"][0] - self.agent_pos[0]) + abs(obj["pos"][1] - self.agent_pos[1])
                    if dist <= 1:
                        obj["open"] = not obj["open"]
                        state = "opened" if obj["open"] else "closed"
                        obs = f"You {state} the {obj['color']} door."
                        break
            else:
                obs = "No door nearby."

        elif action_lower == "done":
            # Check if goal is completed
            done = True
            reward = 1.0 if self._check_goal() else 0.0
            obs = "Task completed!" if reward > 0 else "Task failed."

        else:
            obs = f"Unknown action: {action}"

        # Check for automatic goal completion
        if self._check_goal():
            done = True
            reward = 1.0
            obs += " Goal achieved!"

        progress = min(self.step_count / 20, 0.8) if not done else (1.0 if reward > 0 else 0.4)

        return obs, reward, done, {"success": reward > 0, "progress": progress}

    def _check_goal(self) -> bool:
        """Check if goal is achieved."""
        goal_lower = self.goal.lower()

        # Simple goal checking
        if "pick up" in goal_lower or "get" in goal_lower:
            for color in ["red", "blue", "green", "yellow"]:
                for obj_type in ["ball", "key", "box"]:
                    if color in goal_lower and obj_type in goal_lower:
                        return self.carrying and color in self.carrying and obj_type in self.carrying

        if "go to" in goal_lower:
            for color in ["red", "blue", "green", "yellow"]:
                for obj_type in ["ball", "key", "box", "door"]:
                    if color in goal_lower and obj_type in goal_lower:
                        for name, obj in self.objects.items():
                            if obj["color"] == color and obj["type"] == obj_type:
                                dist = abs(obj["pos"][0] - self.agent_pos[0]) + abs(obj["pos"][1] - self.agent_pos[1])
                                return dist <= 1

        return False


class BabyAIDataset(MultiTurnDataset):
    """
    BabyAI dataset for grounded language navigation.

    Features compositional language instructions in a grid world.
    """

    def __init__(
        self,
        data_path: Optional[str] = None,
        split: DatasetSplit = DatasetSplit.TEST,
        level: Optional[str] = None,  # GoTo, Pickup, Open, etc.
        **kwargs,
    ):
        """
        Initialize BabyAI dataset.

        Args:
            data_path: Path to BabyAI data
            split: Dataset split
            level: Filter by difficulty level
        """
        super().__init__(data_path, split, **kwargs)
        self.level = level

    @property
    def name(self) -> str:
        return "babyai"

    def _load_data(self) -> List[TaskInstance]:
        """Load BabyAI tasks."""
        # Sample BabyAI tasks
        tasks = [
            {"goal": "go to the red ball", "level": "GoTo"},
            {"goal": "pick up the blue key", "level": "Pickup"},
            {"goal": "open the yellow door", "level": "Open"},
            {"goal": "put the green box next to the red ball", "level": "PutNext"},
            {"goal": "go to the blue key then pick it up", "level": "Seq"},
            {"goal": "pick up the red ball or the blue key", "level": "Or"},
            {"goal": "go to the green box and open the yellow door", "level": "And"},
            {"goal": "pick up a ball", "level": "GoToObj"},
            {"goal": "go to an open door", "level": "GoToDoor"},
            {"goal": "pick up the box after you open the door", "level": "SeqS"},
        ]

        instances = []
        for idx, task in enumerate(tasks):
            task_level = task.get("level", "GoTo")

            if self.level and task_level != self.level:
                continue

            instances.append(TaskInstance(
                task_id=f"babyai_{idx}",
                input_text=task["goal"],
                target="success",
                metadata={"level": task_level},
                domain="navigation",
                difficulty=self._estimate_difficulty(task_level),
            ))

        return instances

    def _estimate_difficulty(self, level: str) -> str:
        """Estimate task difficulty by level."""
        easy_levels = ["GoTo", "Pickup"]
        medium_levels = ["Open", "GoToObj", "GoToDoor"]
        hard_levels = ["PutNext", "Seq", "And", "Or", "SeqS"]

        if level in easy_levels:
            return "easy"
        elif level in medium_levels:
            return "medium"
        return "hard"

    def get_environment(self, task_instance: TaskInstance) -> BabyAIEnvironment:
        """Get BabyAI environment for task."""
        return BabyAIEnvironment(task_instance)

    def get_environment_info(self, task_instance: TaskInstance) -> str:
        """Get environment instructions."""
        return """You are in a grid world. Available actions:
- turn left: Turn 90 degrees left
- turn right: Turn 90 degrees right
- go forward: Move one cell forward
- pick up: Pick up object at current position
- drop: Drop carried object
- toggle: Open/close door
- done: Declare task complete"""

    def evaluate(self, prediction: str, target: str) -> Dict[str, Any]:
        """Evaluate based on task completion."""
        return {
            "success": prediction.lower() == "success",
            "progress": 1.0 if prediction.lower() == "success" else 0.0,
        }
