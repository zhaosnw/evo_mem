"""Context builder for memory synthesis.

Implements the Synthesis operation: C̃_t = C(x_t, R_t)
where retrieved information is restructured into working context.
"""

from typing import List, Optional, Dict, Any
from abc import ABC, abstractmethod

from .base import MemoryEntry
from .retriever import RetrievalResult


class ContextBuilder(ABC):
    """
    Abstract base class for context building.

    The context builder implements: C̃_t = C(x_t, R_t)
    """

    @abstractmethod
    def build(
        self,
        query: str,
        retrieved: List[RetrievalResult],
        **kwargs,
    ) -> str:
        """
        Build context from query and retrieved memories.

        Args:
            query: Current input query (x_t)
            retrieved: Retrieved memory entries (R_t)
            **kwargs: Additional context building options

        Returns:
            Constructed context string (C̃_t)
        """
        pass


class SimpleContextBuilder(ContextBuilder):
    """
    Simple context builder that concatenates retrieved experiences.

    Used as a baseline for ExpRAG and other simple methods.
    """

    def __init__(
        self,
        include_trajectory: bool = True,
        include_feedback: bool = True,
        max_context_length: int = 8000,
    ):
        """
        Initialize context builder.

        Args:
            include_trajectory: Include action trajectories
            include_feedback: Include feedback/correctness signals
            max_context_length: Maximum context length in characters
        """
        self.include_trajectory = include_trajectory
        self.include_feedback = include_feedback
        self.max_context_length = max_context_length

    def build(
        self,
        query: str,
        retrieved: List[RetrievalResult],
        **kwargs,
    ) -> str:
        """Build context by concatenating retrieved experiences."""
        if not retrieved:
            return query

        # Build experience section
        experience_parts = []
        for i, result in enumerate(retrieved):
            entry = result.entry
            exp_text = entry.to_text(
                include_trajectory=self.include_trajectory,
                include_feedback=self.include_feedback,
            )
            experience_parts.append(f"[Experience #{i + 1}]\n{exp_text}")

        experiences = "\n\n".join(experience_parts)

        # Truncate if needed
        if len(experiences) > self.max_context_length:
            experiences = experiences[:self.max_context_length] + "..."

        # Combine with query
        context = f"""==================================================
RELEVANT EXPERIENCE FROM SIMILAR TASKS
==================================================
{experiences}

==================================================
YOUR CURRENT TASK
==================================================
{query}"""

        return context


class StructuredContextBuilder(ContextBuilder):
    """
    Structured context builder for multi-turn tasks.

    Follows the prompt template format from the paper's appendix.
    """

    def __init__(
        self,
        include_trajectory: bool = True,
        include_feedback: bool = True,
        max_experiences: int = 4,
        max_history_steps: int = 10,
    ):
        """
        Initialize structured context builder.

        Args:
            include_trajectory: Include action trajectories
            include_feedback: Include feedback signals
            max_experiences: Maximum number of experiences to include
            max_history_steps: Maximum history steps to include
        """
        self.include_trajectory = include_trajectory
        self.include_feedback = include_feedback
        self.max_experiences = max_experiences
        self.max_history_steps = max_history_steps

    def build(
        self,
        query: str,
        retrieved: List[RetrievalResult],
        environment_instructions: str = "",
        example_demonstrations: str = "",
        current_goal: str = "",
        recent_history: Optional[List[Dict[str, str]]] = None,
        **kwargs,
    ) -> str:
        """Build structured context for multi-turn tasks."""
        parts = []

        # Environment instructions
        if environment_instructions:
            parts.append(f"""==================================================
ENVIRONMENT INSTRUCTIONS
==================================================
{environment_instructions}""")

        # Example demonstrations
        if example_demonstrations:
            parts.append(f"""==================================================
EXAMPLE DEMONSTRATIONS
==================================================
{example_demonstrations}""")

        # Retrieved experiences
        if retrieved:
            exp_parts = []
            for i, result in enumerate(retrieved[:self.max_experiences]):
                entry = result.entry
                exp_text = f"""[Experience #{i + 1}]
Goal: {entry.input_text}
Trajectory: {self._format_trajectory(entry.trajectory)}
Correctness: {'Success' if entry.is_successful else 'Failure'}"""
                exp_parts.append(exp_text)

            parts.append(f"""==================================================
RELEVANT EXPERIENCE FROM SIMILAR TASKS
==================================================
{chr(10).join(exp_parts)}""")

        # Current task
        goal_text = current_goal if current_goal else query
        parts.append(f"""==================================================
YOUR CURRENT TASK
==================================================
Goal: {goal_text}
Help: type 'check valid actions' if action fails
Help: type 'inventory' to check items""")

        # Recent history
        if recent_history:
            history_str = self._format_history(recent_history)
            parts.append(f"""==================================================
RECENT HISTORY
==================================================
{history_str}""")

        # Output format
        parts.append("""==================================================
OUTPUT FORMAT
==================================================
You MUST respond in EXACTLY ONE of these formats:

Format 1 - Prune experiences:
Think-Prune: <IDs>
Remove unhelpful experiences from 'RELEVANT EXPERIENCE' section (e.g., "1,3" or "2-4")

Format 2 - Internal reasoning:
Think: <your reasoning>
Free-form explanation of your next step

Format 3 - Execute action:
Action: <exact command>
Must be valid command from ENVIRONMENT INSTRUCTIONS with exact names from RECENT HISTORY""")

        return "\n\n".join(parts)

    def _format_trajectory(self, trajectory: Optional[List[Dict[str, str]]]) -> str:
        """Format trajectory for display."""
        if not trajectory:
            return "N/A"

        steps = []
        for step in trajectory[:self.max_history_steps]:
            action = step.get("action", "")
            obs = step.get("observation", "")
            steps.append(f"{action} -> {obs[:100]}...")

        return " | ".join(steps)

    def _format_history(self, history: List[Dict[str, str]]) -> str:
        """Format recent history for display."""
        lines = []
        for step in history[-self.max_history_steps:]:
            if "observation" in step:
                lines.append(f"Observation: {step['observation']}")
            if "action" in step:
                lines.append(f"Action: {step['action']}")

        return "\n".join(lines)


class SingleTurnContextBuilder(ContextBuilder):
    """
    Context builder for single-turn reasoning tasks.

    Follows the single-turn prompt template from the paper.
    """

    def __init__(
        self,
        include_rationale: bool = True,
        max_experiences: int = 4,
    ):
        """
        Initialize single-turn context builder.

        Args:
            include_rationale: Include rationale in experiences
            max_experiences: Maximum experiences to include
        """
        self.include_rationale = include_rationale
        self.max_experiences = max_experiences

    def build(
        self,
        query: str,
        retrieved: List[RetrievalResult],
        **kwargs,
    ) -> str:
        """Build context for single-turn tasks."""
        # Build memory section
        if retrieved:
            memory_parts = []
            for i, result in enumerate(retrieved[:self.max_experiences]):
                entry = result.entry
                parts = [f"[Memory {i + 1}]"]
                parts.append(f"Question: {entry.input_text[:200]}...")
                parts.append(f"Answer: {entry.output_text[:200]}...")

                if self.include_rationale and entry.metadata.get("rationale"):
                    parts.append(f"Rationale: {entry.metadata['rationale'][:200]}...")

                if entry.metadata.get("domain"):
                    parts.append(f"Domain: {entry.metadata['domain']}")

                memory_parts.append("\n".join(parts))

            memories = "\n\n".join(memory_parts)
        else:
            memories = "No relevant memories found."

        context = f"""You are a helpful assistant with access to LOCAL EXPERIENCE MEMORY. Each memory may contain past experience, rationales, domains, and skills. Below are some retrieved LOCAL EXPERIENCE MEMORIES:

{memories}

Now solve the following problem.
Question: {query}

Provide your output in the following format:
- Rationale: your short reasoning, may cite memory if useful
- Final Answer: your final answer"""

        return context


class CheatsheetContextBuilder(ContextBuilder):
    """
    Context builder for Dynamic Cheatsheet method.

    Maintains a cumulative or synthesized cheatsheet of strategies.
    """

    def __init__(self, mode: str = "cumulative"):
        """
        Initialize cheatsheet context builder.

        Args:
            mode: Either "cumulative" (DC-Cu) or "synthesis" (DC-RS)
        """
        self.mode = mode
        self.cheatsheet: List[str] = []

    def add_to_cheatsheet(self, strategy: str) -> None:
        """Add a new strategy to the cheatsheet."""
        if strategy not in self.cheatsheet:
            self.cheatsheet.append(strategy)

    def synthesize_cheatsheet(self, llm_synthesize_fn) -> str:
        """Synthesize cheatsheet entries into a coherent summary."""
        if not self.cheatsheet:
            return ""

        if self.mode == "cumulative":
            return "\n".join([f"- {s}" for s in self.cheatsheet])
        else:
            # Use LLM to synthesize
            combined = "\n".join(self.cheatsheet)
            return llm_synthesize_fn(combined)

    def build(
        self,
        query: str,
        retrieved: List[RetrievalResult],
        **kwargs,
    ) -> str:
        """Build context with cheatsheet."""
        cheatsheet_text = self.synthesize_cheatsheet(
            kwargs.get("llm_synthesize_fn", lambda x: x)
        )

        if cheatsheet_text:
            context = f"""==================================================
STRATEGY CHEATSHEET
==================================================
{cheatsheet_text}

==================================================
CURRENT TASK
==================================================
{query}"""
        else:
            context = query

        return context

    def clear(self) -> None:
        """Clear the cheatsheet."""
        self.cheatsheet.clear()
