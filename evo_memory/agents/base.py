"""Base agent class for Evo-Memory.

This module implements the core agent abstraction that follows the
search-synthesize-evolve loop described in the paper.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional, Tuple
from enum import Enum
import re

from ..memory import Memory, MemoryEntry, Retriever, ContextBuilder
from ..memory.retriever import RetrievalResult
from ..llm import BaseLLM, LLMResponse


class ActionType(Enum):
    """Types of agent actions."""
    THINK = "think"      # Internal reasoning
    ACT = "act"          # Execute action / produce output
    REFINE = "refine"    # Memory refinement (prune, organize)


@dataclass
class AgentAction:
    """Represents an action taken by the agent."""
    action_type: ActionType
    content: str
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class AgentState:
    """Current state of the agent."""
    task_id: str
    input_text: str
    memory: Memory
    retrieved: List[RetrievalResult] = field(default_factory=list)
    action_history: List[AgentAction] = field(default_factory=list)
    observations: List[str] = field(default_factory=list)
    current_step: int = 0
    is_complete: bool = False
    final_output: Optional[str] = None
    feedback: Optional[str] = None
    is_successful: bool = False


class BaseAgent(ABC):
    """
    Abstract base class for memory-augmented agents.

    Implements the general memory-augmented agent tuple (F, U, R, C):
    - F: Base LLM (self.llm)
    - U: Memory update pipeline (self.update_memory)
    - R: Retrieval module (self.retriever)
    - C: Contextual construction (self.context_builder)
    """

    def __init__(
        self,
        llm: BaseLLM,
        retriever: Retriever,
        context_builder: ContextBuilder,
        memory: Optional[Memory] = None,
        top_k: int = 4,
        max_steps: int = 50,
        max_iterations: int = 10,
        store_successful_only: bool = True,
    ):
        """
        Initialize agent.

        Args:
            llm: Base LLM for generation
            retriever: Retriever for memory search
            context_builder: Context builder for synthesis
            memory: Optional pre-initialized memory
            top_k: Number of memories to retrieve
            max_steps: Maximum steps per task (multi-turn)
            max_iterations: Maximum Think/Refine iterations per step
            store_successful_only: Only store successful experiences
        """
        self.llm = llm
        self.retriever = retriever
        self.context_builder = context_builder
        self.memory = memory or Memory()
        self.top_k = top_k
        self.max_steps = max_steps
        self.max_iterations = max_iterations
        self.store_successful_only = store_successful_only

        # Statistics
        self.total_tasks = 0
        self.successful_tasks = 0

    def search(self, query: str) -> List[RetrievalResult]:
        """
        Search memory for relevant experiences.

        Implements: R_t = R(M_t, x_t)

        Args:
            query: Current task input (x_t)

        Returns:
            List of retrieved memory entries (R_t)
        """
        return self.retriever.retrieve(
            query=query,
            memory=self.memory,
            top_k=self.top_k,
        )

    def synthesize(
        self,
        query: str,
        retrieved: List[RetrievalResult],
        **kwargs,
    ) -> str:
        """
        Synthesize context from query and retrieved memories.

        Implements: C̃_t = C(x_t, R_t)

        Args:
            query: Current task input (x_t)
            retrieved: Retrieved memories (R_t)
            **kwargs: Additional context building parameters

        Returns:
            Synthesized context (C̃_t)
        """
        return self.context_builder.build(
            query=query,
            retrieved=retrieved,
            **kwargs,
        )

    def evolve(self, state: AgentState) -> None:
        """
        Update memory with current experience.

        Implements: M_{t+1} = U(M_t, m_t)

        Args:
            state: Current agent state containing experience
        """
        # Check if we should store this experience
        if self.store_successful_only and not state.is_successful:
            return

        # Create memory entry
        entry = MemoryEntry(
            task_id=state.task_id,
            input_text=state.input_text,
            output_text=state.final_output or "",
            feedback=state.feedback,
            trajectory=[
                {"action": a.content, "observation": o}
                for a, o in zip(
                    [a for a in state.action_history if a.action_type == ActionType.ACT],
                    state.observations,
                )
            ] if state.observations else None,
            is_successful=state.is_successful,
        )

        # Compute embedding
        text = entry.to_text()
        entry.embedding = self.retriever.encode(text)

        # Add to memory
        self.memory.add(entry)

    @abstractmethod
    def run_single_turn(
        self,
        task_id: str,
        query: str,
        **kwargs,
    ) -> Tuple[str, AgentState]:
        """
        Run agent on a single-turn task.

        Args:
            task_id: Unique task identifier
            query: Task query/question

        Returns:
            Tuple of (output, final_state)
        """
        pass

    @abstractmethod
    def run_multi_turn(
        self,
        task_id: str,
        goal: str,
        environment,
        **kwargs,
    ) -> Tuple[bool, float, AgentState]:
        """
        Run agent on a multi-turn task.

        Args:
            task_id: Unique task identifier
            goal: Task goal description
            environment: Environment interface for actions

        Returns:
            Tuple of (success, progress, final_state)
        """
        pass

    def parse_response(self, response: str) -> AgentAction:
        """
        Parse LLM response into an agent action.

        Default implementation for simple action parsing.
        Override for more complex parsing.
        """
        response = response.strip()

        # Check for Think-Prune (memory refinement)
        prune_match = re.match(r"Think-Prune:\s*(.+)", response, re.IGNORECASE)
        if prune_match:
            return AgentAction(
                action_type=ActionType.REFINE,
                content=prune_match.group(1).strip(),
                metadata={"type": "prune"},
            )

        # Check for Think
        think_match = re.match(r"Think:\s*(.+)", response, re.IGNORECASE | re.DOTALL)
        if think_match:
            return AgentAction(
                action_type=ActionType.THINK,
                content=think_match.group(1).strip(),
            )

        # Check for Action
        action_match = re.match(r"Action:\s*(.+)", response, re.IGNORECASE | re.DOTALL)
        if action_match:
            return AgentAction(
                action_type=ActionType.ACT,
                content=action_match.group(1).strip(),
            )

        # Default: treat as action
        return AgentAction(
            action_type=ActionType.ACT,
            content=response,
        )

    def extract_answer(self, response: str) -> str:
        """
        Extract final answer from response.

        Looks for "Final Answer:" pattern.
        """
        # Try to find explicit final answer
        match = re.search(
            r"Final Answer:\s*(.+?)(?:\n|$)",
            response,
            re.IGNORECASE | re.DOTALL,
        )
        if match:
            return match.group(1).strip()

        # Try to find answer in common patterns
        for pattern in [
            r"The answer is\s*[:.]?\s*(.+?)(?:\n|$)",
            r"Answer:\s*(.+?)(?:\n|$)",
            r"Therefore[,]?\s*(.+?)(?:\n|$)",
        ]:
            match = re.search(pattern, response, re.IGNORECASE | re.DOTALL)
            if match:
                return match.group(1).strip()

        # Return the whole response if no pattern matches
        return response.strip()

    def get_statistics(self) -> Dict[str, Any]:
        """Get agent statistics."""
        memory_stats = self.memory.get_statistics()
        llm_stats = self.llm.get_usage_stats()

        return {
            "total_tasks": self.total_tasks,
            "successful_tasks": self.successful_tasks,
            "success_rate": self.successful_tasks / max(self.total_tasks, 1),
            "memory": memory_stats,
            "llm_usage": llm_stats,
        }

    def reset_memory(self) -> None:
        """Reset agent memory."""
        self.memory.clear()

    def reset_statistics(self) -> None:
        """Reset all statistics."""
        self.total_tasks = 0
        self.successful_tasks = 0
        self.llm.reset_usage_stats()
