"""Agent Workflow Memory (AWM) implementation.

AWM emphasizes reusable workflows and task strategies,
organizing experiences into procedural forms.

Based on Wang et al. (2024) as referenced in the paper.
"""

from typing import Tuple, Optional, List, Dict, Any
from dataclasses import dataclass, field
import re

from .base import BaseAgent, AgentState, AgentAction, ActionType
from ..memory import Memory, MemoryEntry, Retriever, ContextBuilder
from ..memory.retriever import RetrievalResult
from ..memory.context import SimpleContextBuilder
from ..llm import BaseLLM


@dataclass
class Workflow:
    """Represents a reusable workflow."""
    id: str
    name: str
    description: str
    steps: List[str]
    applicable_goals: List[str] = field(default_factory=list)
    success_count: int = 0
    usage_count: int = 0

    @property
    def success_rate(self) -> float:
        return self.success_count / max(self.usage_count, 1)

    def to_text(self) -> str:
        steps_str = "\n".join([f"  {i+1}. {s}" for i, s in enumerate(self.steps)])
        return f"{self.name}: {self.description}\nSteps:\n{steps_str}"


class WorkflowLibrary:
    """Library of reusable workflows."""

    def __init__(self, max_size: int = 50):
        self.workflows: Dict[str, Workflow] = {}
        self.max_size = max_size

    def add(self, workflow: Workflow) -> None:
        """Add a workflow to the library."""
        if len(self.workflows) >= self.max_size:
            # Remove least successful workflow
            worst = min(
                self.workflows.values(),
                key=lambda w: w.success_rate * w.usage_count
            )
            del self.workflows[worst.id]

        self.workflows[workflow.id] = workflow

    def search(self, goal: str, top_k: int = 3) -> List[Workflow]:
        """Search for relevant workflows."""
        if not self.workflows:
            return []

        goal_words = set(goal.lower().split())
        scored = []

        for workflow in self.workflows.values():
            # Score based on goal similarity and success rate
            desc_words = set(workflow.description.lower().split())
            name_words = set(workflow.name.lower().split())
            applicable_words = set(" ".join(workflow.applicable_goals).lower().split())

            overlap = len(goal_words & (desc_words | name_words | applicable_words))
            score = overlap * (1 + workflow.success_rate)

            if overlap > 0:
                scored.append((workflow, score))

        scored.sort(key=lambda x: x[1], reverse=True)
        return [w for w, _ in scored[:top_k]]

    def update_success(self, workflow_id: str, success: bool) -> None:
        """Update workflow success statistics."""
        if workflow_id in self.workflows:
            self.workflows[workflow_id].usage_count += 1
            if success:
                self.workflows[workflow_id].success_count += 1

    def clear(self) -> None:
        """Clear all workflows."""
        self.workflows.clear()


class AWMAgent(BaseAgent):
    """
    Agent Workflow Memory (AWM).

    Organizes experiences into reusable workflows that capture
    procedural knowledge for accomplishing goals.
    """

    def __init__(
        self,
        llm: BaseLLM,
        retriever: Retriever,
        context_builder: Optional[ContextBuilder] = None,
        memory: Optional[Memory] = None,
        workflow_library: Optional[WorkflowLibrary] = None,
        min_workflow_steps: int = 2,
        max_workflow_steps: int = 10,
        **kwargs,
    ):
        """
        Initialize AWM agent.

        Args:
            llm: Base LLM for generation
            retriever: Retriever for experience search
            context_builder: Optional context builder
            memory: Optional experience memory
            workflow_library: Optional workflow library
            min_workflow_steps: Minimum steps to create workflow
            max_workflow_steps: Maximum steps in a workflow
        """
        context_builder = context_builder or SimpleContextBuilder()

        super().__init__(
            llm=llm,
            retriever=retriever,
            context_builder=context_builder,
            memory=memory,
            **kwargs,
        )

        self.workflow_library = workflow_library or WorkflowLibrary()
        self.min_workflow_steps = min_workflow_steps
        self.max_workflow_steps = max_workflow_steps
        self.current_workflow: Optional[Workflow] = None

    def run_single_turn(
        self,
        task_id: str,
        query: str,
        **kwargs,
    ) -> Tuple[str, AgentState]:
        """Run AWM on a single-turn task."""
        self.total_tasks += 1

        state = AgentState(
            task_id=task_id,
            input_text=query,
            memory=self.memory,
        )

        # Search for relevant workflows
        workflows = self.workflow_library.search(query, top_k=2)

        # Search experience memory
        if self.memory and len(self.memory) > 0:
            state.retrieved = self.search(query)

        # Build context
        context = self._build_context(query, workflows, state.retrieved)

        # Generate response
        response = self.llm.generate(prompt=context)
        output = self.extract_answer(response.content)

        # Update state
        state.final_output = output
        state.is_complete = True
        state.feedback = kwargs.get("feedback")
        state.is_successful = kwargs.get("is_correct", False)

        if state.is_successful:
            self.successful_tasks += 1

        self.evolve(state)
        return output, state

    def run_multi_turn(
        self,
        task_id: str,
        goal: str,
        environment,
        **kwargs,
    ) -> Tuple[bool, float, AgentState]:
        """Run AWM on a multi-turn task."""
        self.total_tasks += 1

        state = AgentState(
            task_id=task_id,
            input_text=goal,
            memory=self.memory,
        )

        observation = environment.reset()
        state.observations.append(observation)

        # Search for relevant workflows
        workflows = self.workflow_library.search(goal, top_k=3)

        # Select best matching workflow
        self.current_workflow = workflows[0] if workflows else None

        # Search experience memory
        if self.memory and len(self.memory) > 0:
            state.retrieved = self.search(goal)

        success = False
        progress = 0.0
        workflow_step_idx = 0

        for step in range(self.max_steps):
            state.current_step = step

            # Get workflow guidance
            workflow_guidance = self._get_workflow_guidance(workflow_step_idx)

            context = self._build_multi_turn_context(
                goal=goal,
                state=state,
                workflows=workflows,
                workflow_guidance=workflow_guidance,
                current_observation=observation,
                environment_info=kwargs.get("environment_info", ""),
            )

            response = self.llm.generate(prompt=context)
            action = self.parse_response(response.content)
            state.action_history.append(action)

            if action.action_type == ActionType.ACT:
                observation, reward, done, info = environment.step(action.content)
                state.observations.append(observation)
                workflow_step_idx += 1

                if "progress" in info:
                    progress = info["progress"]

                if done:
                    success = info.get("success", False)
                    break

        # Update workflow statistics
        if self.current_workflow:
            self.workflow_library.update_success(self.current_workflow.id, success)

        # Extract new workflow from successful task
        if success and len(state.action_history) >= self.min_workflow_steps:
            new_workflow = self._extract_workflow(goal, state.action_history)
            if new_workflow:
                self.workflow_library.add(new_workflow)

        state.is_complete = True
        state.is_successful = success
        state.feedback = "Success" if success else "Failure"
        state.final_output = state.action_history[-1].content if state.action_history else ""

        if success:
            self.successful_tasks += 1

        self.evolve(state)
        return success, progress, state

    def _get_workflow_guidance(self, step_idx: int) -> Optional[str]:
        """Get guidance from current workflow."""
        if not self.current_workflow:
            return None

        if step_idx < len(self.current_workflow.steps):
            return self.current_workflow.steps[step_idx]

        return None

    def _extract_workflow(
        self,
        goal: str,
        action_history: List[AgentAction],
    ) -> Optional[Workflow]:
        """Extract a workflow from successful task execution."""
        actions = [
            a.content for a in action_history
            if a.action_type == ActionType.ACT
        ][:self.max_workflow_steps]

        if len(actions) < self.min_workflow_steps:
            return None

        # Generate workflow name and description
        prompt = f"""Create a reusable workflow from this successful task.

Goal: {goal}
Actions taken: {' -> '.join(actions)}

Provide:
Name: [short descriptive name]
Description: [one sentence description]"""

        response = self.llm.generate(prompt=prompt, max_tokens=100)

        # Parse response
        name_match = re.search(r"Name:\s*(.+?)(?:\n|$)", response.content)
        desc_match = re.search(r"Description:\s*(.+?)(?:\n|$)", response.content)

        name = name_match.group(1).strip() if name_match else f"Workflow for {goal[:30]}..."
        description = desc_match.group(1).strip() if desc_match else goal

        import hashlib
        workflow_id = hashlib.md5(f"{goal}:{':'.join(actions)}".encode()).hexdigest()[:8]

        return Workflow(
            id=workflow_id,
            name=name,
            description=description,
            steps=actions,
            applicable_goals=[goal[:100]],
            success_count=1,
            usage_count=1,
        )

    def _build_context(
        self,
        query: str,
        workflows: List[Workflow],
        retrieved: List[RetrievalResult],
    ) -> str:
        """Build context for single-turn tasks."""
        parts = []

        if workflows:
            workflow_str = "\n\n".join([w.to_text() for w in workflows[:2]])
            parts.append(f"Relevant Workflows:\n{workflow_str}")

        if retrieved:
            exp_str = "\n".join([f"- {r.entry.output_text[:100]}..." for r in retrieved[:3]])
            parts.append(f"Past Examples:\n{exp_str}")

        parts.append(f"Question: {query}")
        parts.append("Answer:")

        return "\n\n".join(parts)

    def _build_multi_turn_context(
        self,
        goal: str,
        state: AgentState,
        workflows: List[Workflow],
        workflow_guidance: Optional[str],
        current_observation: str,
        environment_info: str = "",
    ) -> str:
        """Build context for multi-turn tasks."""
        parts = []

        if environment_info:
            parts.append(f"Environment: {environment_info}")

        if workflows:
            workflow_summaries = [f"- {w.name}: {w.description}" for w in workflows[:3]]
            parts.append("Available Workflows:\n" + "\n".join(workflow_summaries))

        if workflow_guidance:
            parts.append(f"Suggested Next Step: {workflow_guidance}")

        if state.retrieved:
            exp_parts = [f"- {r.entry.input_text[:60]}..." for r in state.retrieved[:2]]
            parts.append("Similar Tasks:\n" + "\n".join(exp_parts))

        parts.append(f"Goal: {goal}")

        if state.observations:
            history = [f"Obs: {obs}" for obs in state.observations[-3:]]
            parts.append("Recent:\n" + "\n".join(history))

        parts.append(f"Current: {current_observation}")
        parts.append("Action:")

        return "\n\n".join(parts)

    def get_workflows(self) -> List[Workflow]:
        """Get all workflows in the library."""
        return list(self.workflow_library.workflows.values())

    def clear_workflows(self) -> None:
        """Clear the workflow library."""
        self.workflow_library.clear()
