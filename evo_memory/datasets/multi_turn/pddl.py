"""PDDL dataset loader.

PDDL (Planning Domain Definition Language) tasks for symbolic planning,
testing the agent's ability to reason about states and actions.
"""

from typing import List, Dict, Any, Optional, Tuple, Set
import json
from dataclasses import dataclass, field

from ..base import MultiTurnDataset, TaskInstance, DatasetSplit


@dataclass
class PDDLState:
    """PDDL state representation."""
    predicates: Set[str] = field(default_factory=set)

    def satisfies(self, goal: Set[str]) -> bool:
        """Check if state satisfies goal."""
        return goal.issubset(self.predicates)

    def copy(self) -> "PDDLState":
        return PDDLState(predicates=self.predicates.copy())


class PDDLEnvironment:
    """
    Simulated PDDL environment.

    Implements a simplified Blocksworld domain.
    """

    def __init__(self, task: TaskInstance):
        self.task = task
        self.goal_predicates = self._parse_goal(task.input_text)
        self.state = self._initialize_state(task.metadata)
        self.step_count = 0
        self.max_steps = 30

    def _parse_goal(self, goal_text: str) -> Set[str]:
        """Parse goal from text."""
        goal_text = goal_text.lower()
        predicates = set()

        # Parse on(X, Y) patterns
        import re
        on_matches = re.findall(r'(\w+) on (\w+)', goal_text)
        for obj, surface in on_matches:
            predicates.add(f"on({obj},{surface})")

        # Parse clear(X) patterns
        clear_matches = re.findall(r'(\w+) is clear', goal_text)
        for obj in clear_matches:
            predicates.add(f"clear({obj})")

        # Parse holding(X) patterns
        holding_matches = re.findall(r'holding (\w+)', goal_text)
        for obj in holding_matches:
            predicates.add(f"holding({obj})")

        return predicates

    def _initialize_state(self, metadata: Dict) -> PDDLState:
        """Initialize PDDL state."""
        init_predicates = metadata.get("init_state", set())
        if isinstance(init_predicates, list):
            init_predicates = set(init_predicates)
        elif not init_predicates:
            # Default Blocksworld initial state
            init_predicates = {
                "on(a,table)",
                "on(b,table)",
                "on(c,a)",
                "clear(b)",
                "clear(c)",
                "clear(table)",
                "arm-empty",
            }
        return PDDLState(predicates=init_predicates)

    def reset(self) -> str:
        """Reset environment."""
        self.step_count = 0
        self.state = self._initialize_state(self.task.metadata)
        return self._get_observation()

    def _get_observation(self) -> str:
        """Get current state observation."""
        return f"Current state: {', '.join(sorted(self.state.predicates))}. Goal: {', '.join(sorted(self.goal_predicates))}"

    def step(self, action: str) -> Tuple[str, float, bool, Dict]:
        """Execute PDDL action."""
        self.step_count += 1
        action = action.lower().strip()

        if self.step_count >= self.max_steps:
            return "Max steps reached.", 0.0, True, {"success": False, "progress": 0.3}

        reward = 0.0
        done = False
        obs = "Action failed."

        # Parse action
        import re

        # Pick-up action: pickup(X)
        pickup_match = re.match(r'pickup\s*\(?(\w+)\)?', action)
        if pickup_match:
            obj = pickup_match.group(1)
            # Preconditions: clear(obj), on(obj,?), arm-empty
            if (f"clear({obj})" in self.state.predicates and
                "arm-empty" in self.state.predicates):
                # Find what obj is on
                for pred in list(self.state.predicates):
                    if pred.startswith(f"on({obj},"):
                        surface = pred.split(",")[1].rstrip(")")
                        self.state.predicates.remove(pred)
                        self.state.predicates.add(f"clear({surface})")
                        break

                self.state.predicates.remove("arm-empty")
                self.state.predicates.remove(f"clear({obj})")
                self.state.predicates.add(f"holding({obj})")
                obs = f"Picked up {obj}."
                reward = 0.1

        # Put-down action: putdown(X)
        putdown_match = re.match(r'putdown\s*\(?(\w+)\)?', action)
        if putdown_match:
            obj = putdown_match.group(1)
            if f"holding({obj})" in self.state.predicates:
                self.state.predicates.remove(f"holding({obj})")
                self.state.predicates.add(f"on({obj},table)")
                self.state.predicates.add("arm-empty")
                self.state.predicates.add(f"clear({obj})")
                obs = f"Put {obj} on table."
                reward = 0.1

        # Stack action: stack(X,Y)
        stack_match = re.match(r'stack\s*\(?(\w+)\s*,\s*(\w+)\)?', action)
        if stack_match:
            obj, dest = stack_match.groups()
            if (f"holding({obj})" in self.state.predicates and
                f"clear({dest})" in self.state.predicates):
                self.state.predicates.remove(f"holding({obj})")
                self.state.predicates.remove(f"clear({dest})")
                self.state.predicates.add(f"on({obj},{dest})")
                self.state.predicates.add("arm-empty")
                self.state.predicates.add(f"clear({obj})")
                obs = f"Stacked {obj} on {dest}."
                reward = 0.2

        # Unstack action: unstack(X,Y)
        unstack_match = re.match(r'unstack\s*\(?(\w+)\s*,\s*(\w+)\)?', action)
        if unstack_match:
            obj, surface = unstack_match.groups()
            if (f"on({obj},{surface})" in self.state.predicates and
                f"clear({obj})" in self.state.predicates and
                "arm-empty" in self.state.predicates):
                self.state.predicates.remove(f"on({obj},{surface})")
                self.state.predicates.remove(f"clear({obj})")
                self.state.predicates.remove("arm-empty")
                self.state.predicates.add(f"clear({surface})")
                self.state.predicates.add(f"holding({obj})")
                obs = f"Unstacked {obj} from {surface}."
                reward = 0.1

        # Check goal
        if self.state.satisfies(self.goal_predicates):
            done = True
            reward = 1.0
            obs += " Goal achieved!"

        progress = len(self.goal_predicates & self.state.predicates) / max(len(self.goal_predicates), 1)

        return obs, reward, done, {"success": done and reward > 0, "progress": progress}


class PDDLDataset(MultiTurnDataset):
    """
    PDDL dataset for symbolic planning.

    Uses Blocksworld domain by default.
    """

    def __init__(
        self,
        data_path: Optional[str] = None,
        split: DatasetSplit = DatasetSplit.TEST,
        domain: str = "blocksworld",
        **kwargs,
    ):
        """
        Initialize PDDL dataset.

        Args:
            data_path: Path to PDDL data
            split: Dataset split
            domain: PDDL domain (blocksworld, logistics, etc.)
        """
        super().__init__(data_path, split, **kwargs)
        self.domain = domain

    @property
    def name(self) -> str:
        return "pddl"

    def _load_data(self) -> List[TaskInstance]:
        """Load PDDL tasks."""
        # Sample Blocksworld tasks
        tasks = [
            {
                "goal": "a on b, b on table",
                "init_state": ["on(a,table)", "on(b,table)", "clear(a)", "clear(b)", "arm-empty"],
            },
            {
                "goal": "a on b, b on c, c on table",
                "init_state": ["on(a,table)", "on(b,table)", "on(c,table)", "clear(a)", "clear(b)", "clear(c)", "arm-empty"],
            },
            {
                "goal": "c on b, b on a",
                "init_state": ["on(a,table)", "on(b,a)", "on(c,table)", "clear(b)", "clear(c)", "arm-empty"],
            },
            {
                "goal": "a on table, b on table, c on table",
                "init_state": ["on(a,b)", "on(b,c)", "on(c,table)", "clear(a)", "arm-empty"],
            },
            {
                "goal": "holding a",
                "init_state": ["on(a,table)", "clear(a)", "arm-empty"],
            },
        ]

        instances = []
        for idx, task in enumerate(tasks):
            instances.append(TaskInstance(
                task_id=f"pddl_{idx}",
                input_text=task["goal"],
                target="success",
                metadata={
                    "init_state": task["init_state"],
                    "domain": self.domain,
                },
                domain="planning",
                difficulty=self._estimate_difficulty(task),
            ))

        return instances

    def _estimate_difficulty(self, task: Dict) -> str:
        """Estimate task difficulty."""
        num_goals = task["goal"].count(",") + 1
        if num_goals <= 1:
            return "easy"
        elif num_goals <= 2:
            return "medium"
        return "hard"

    def get_environment(self, task_instance: TaskInstance) -> PDDLEnvironment:
        """Get PDDL environment for task."""
        return PDDLEnvironment(task_instance)

    def get_environment_info(self, task_instance: TaskInstance) -> str:
        """Get environment instructions."""
        return """This is a Blocksworld planning domain. Available actions:
- pickup(X): Pick up block X from table (requires: clear(X), arm-empty)
- putdown(X): Put held block X on table (requires: holding(X))
- stack(X,Y): Stack block X on block Y (requires: holding(X), clear(Y))
- unstack(X,Y): Remove block X from block Y (requires: on(X,Y), clear(X), arm-empty)

State predicates: on(X,Y), clear(X), holding(X), arm-empty"""

    def evaluate(self, prediction: str, target: str) -> Dict[str, Any]:
        """Evaluate based on goal achievement."""
        return {
            "success": prediction.lower() == "success",
            "progress": 1.0 if prediction.lower() == "success" else 0.0,
        }
