"""Mem0 Agent implementation.

Mem0 implements structured, agent-level memory systems that support
read, write, and update operations.

Based on Chhikara et al. (2025) as referenced in the paper.
"""

from typing import Tuple, Optional, List, Dict, Any
from dataclasses import dataclass, field
import hashlib
import json

from .base import BaseAgent, AgentState, AgentAction, ActionType
from ..memory import Memory, MemoryEntry, Retriever, ContextBuilder
from ..memory.retriever import RetrievalResult
from ..memory.context import SimpleContextBuilder
from ..llm import BaseLLM


@dataclass
class Mem0Entry:
    """Structured memory entry for Mem0."""
    id: str
    content: str
    category: str = "general"
    importance: float = 0.5
    access_count: int = 0
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_text(self) -> str:
        return f"[{self.category}] {self.content}"


class Mem0Store:
    """
    Mem0-style memory store with structured operations.

    Supports:
    - ADD: Add new memory
    - UPDATE: Update existing memory
    - DELETE: Remove memory
    - SEARCH: Find relevant memories
    """

    def __init__(self, max_size: int = 100):
        self.memories: Dict[str, Mem0Entry] = {}
        self.max_size = max_size

    def add(self, content: str, category: str = "general", metadata: Dict = None) -> str:
        """Add a new memory entry."""
        mem_id = hashlib.md5(content.encode()).hexdigest()[:8]

        if len(self.memories) >= self.max_size:
            # Remove least important memory
            least_important = min(
                self.memories.values(),
                key=lambda m: m.importance * (m.access_count + 1)
            )
            del self.memories[least_important.id]

        self.memories[mem_id] = Mem0Entry(
            id=mem_id,
            content=content,
            category=category,
            metadata=metadata or {},
        )
        return mem_id

    def update(self, mem_id: str, content: str = None, importance: float = None) -> bool:
        """Update an existing memory."""
        if mem_id not in self.memories:
            return False

        if content:
            self.memories[mem_id].content = content
        if importance is not None:
            self.memories[mem_id].importance = importance

        return True

    def delete(self, mem_id: str) -> bool:
        """Delete a memory."""
        if mem_id in self.memories:
            del self.memories[mem_id]
            return True
        return False

    def search(self, query: str, category: str = None, top_k: int = 5) -> List[Mem0Entry]:
        """Search memories (simple keyword matching for now)."""
        results = []
        query_lower = query.lower()

        for mem in self.memories.values():
            if category and mem.category != category:
                continue

            # Simple relevance score based on keyword overlap
            content_lower = mem.content.lower()
            overlap = sum(1 for word in query_lower.split() if word in content_lower)
            score = overlap / max(len(query_lower.split()), 1)

            if score > 0:
                mem.access_count += 1
                results.append((mem, score))

        results.sort(key=lambda x: x[1], reverse=True)
        return [m for m, _ in results[:top_k]]

    def get_all(self, category: str = None) -> List[Mem0Entry]:
        """Get all memories, optionally filtered by category."""
        if category:
            return [m for m in self.memories.values() if m.category == category]
        return list(self.memories.values())

    def clear(self) -> None:
        """Clear all memories."""
        self.memories.clear()


class Mem0Agent(BaseAgent):
    """
    Mem0 Agent: Structured agent-level memory.

    Features structured memory with categories, importance scores,
    and explicit memory operations (add, update, delete).
    """

    def __init__(
        self,
        llm: BaseLLM,
        retriever: Retriever,
        context_builder: Optional[ContextBuilder] = None,
        memory: Optional[Memory] = None,
        mem0_store: Optional[Mem0Store] = None,
        top_k: int = 4,
        **kwargs,
    ):
        """
        Initialize Mem0 agent.

        Args:
            llm: Base LLM for generation
            retriever: Retriever for experience-based memory
            context_builder: Optional context builder
            memory: Optional experience memory
            mem0_store: Optional Mem0 structured store
            top_k: Number of items to retrieve
        """
        context_builder = context_builder or SimpleContextBuilder()

        super().__init__(
            llm=llm,
            retriever=retriever,
            context_builder=context_builder,
            memory=memory,
            top_k=top_k,
            **kwargs,
        )

        self.mem0_store = mem0_store or Mem0Store()

    def run_single_turn(
        self,
        task_id: str,
        query: str,
        **kwargs,
    ) -> Tuple[str, AgentState]:
        """Run Mem0 on a single-turn task."""
        self.total_tasks += 1

        state = AgentState(
            task_id=task_id,
            input_text=query,
            memory=self.memory,
        )

        # Search both memory systems
        if self.memory and len(self.memory) > 0:
            state.retrieved = self.search(query)

        mem0_results = self.mem0_store.search(query, top_k=self.top_k)

        # Build context
        context = self._build_context(query, state.retrieved, mem0_results)

        # Generate response
        response = self.llm.generate(prompt=context)
        output = self.extract_answer(response.content)

        # Update Mem0 store
        self._update_mem0_store(query, output, kwargs.get("is_correct", False))

        # Update state
        state.final_output = output
        state.is_complete = True
        state.feedback = kwargs.get("feedback")
        state.is_successful = kwargs.get("is_correct", False)

        if state.is_successful:
            self.successful_tasks += 1

        # Evolve experience memory
        self.evolve(state)

        return output, state

    def run_multi_turn(
        self,
        task_id: str,
        goal: str,
        environment,
        **kwargs,
    ) -> Tuple[bool, float, AgentState]:
        """Run Mem0 on a multi-turn task."""
        self.total_tasks += 1

        state = AgentState(
            task_id=task_id,
            input_text=goal,
            memory=self.memory,
        )

        observation = environment.reset()
        state.observations.append(observation)

        # Search memories
        if self.memory and len(self.memory) > 0:
            state.retrieved = self.search(goal)

        mem0_results = self.mem0_store.search(goal, top_k=self.top_k)

        success = False
        progress = 0.0

        for step in range(self.max_steps):
            state.current_step = step

            context = self._build_multi_turn_context(
                goal=goal,
                state=state,
                mem0_results=mem0_results,
                current_observation=observation,
                environment_info=kwargs.get("environment_info", ""),
            )

            response = self.llm.generate(prompt=context)
            action = self.parse_response(response.content)
            state.action_history.append(action)

            if action.action_type == ActionType.ACT:
                observation, reward, done, info = environment.step(action.content)
                state.observations.append(observation)

                if "progress" in info:
                    progress = info["progress"]

                if done:
                    success = info.get("success", False)
                    break

        # Update Mem0 store with task result
        self._update_mem0_store(goal, str(success), success)

        # Final state
        state.is_complete = True
        state.is_successful = success
        state.feedback = "Success" if success else "Failure"
        state.final_output = state.action_history[-1].content if state.action_history else ""

        if success:
            self.successful_tasks += 1

        self.evolve(state)
        return success, progress, state

    def _build_context(
        self,
        query: str,
        retrieved: List[RetrievalResult],
        mem0_results: List[Mem0Entry],
    ) -> str:
        """Build context combining both memory types."""
        parts = []

        # Mem0 structured memories
        if mem0_results:
            mem_parts = []
            for m in mem0_results:
                mem_parts.append(f"- [{m.category}] {m.content}")
            parts.append("Structured Memories:\n" + "\n".join(mem_parts))

        # Experience memories
        if retrieved:
            exp_parts = []
            for i, r in enumerate(retrieved):
                exp_parts.append(f"[Exp {i+1}] {r.entry.output_text[:150]}...")
            parts.append("Past Experiences:\n" + "\n".join(exp_parts))

        parts.append(f"Question: {query}")
        parts.append("Answer:")

        return "\n\n".join(parts)

    def _build_multi_turn_context(
        self,
        goal: str,
        state: AgentState,
        mem0_results: List[Mem0Entry],
        current_observation: str,
        environment_info: str = "",
    ) -> str:
        """Build context for multi-turn tasks."""
        parts = []

        if environment_info:
            parts.append(f"Environment: {environment_info}")

        if mem0_results:
            mem_parts = [f"- [{m.category}] {m.content}" for m in mem0_results[:3]]
            parts.append("Memories:\n" + "\n".join(mem_parts))

        if state.retrieved:
            exp_parts = [f"[Exp] {r.entry.input_text[:80]}..." for r in state.retrieved[:2]]
            parts.append("Similar Tasks:\n" + "\n".join(exp_parts))

        parts.append(f"Goal: {goal}")

        if state.observations:
            history = [f"Obs: {obs}" for obs in state.observations[-3:]]
            parts.append("Recent:\n" + "\n".join(history))

        parts.append(f"Current: {current_observation}")
        parts.append("Action:")

        return "\n\n".join(parts)

    def _update_mem0_store(self, query: str, output: str, is_correct: bool) -> None:
        """Update Mem0 store based on task result."""
        if is_correct:
            # Extract key insight
            insight = f"For '{query[:50]}...': {output[:100]}..."
            self.mem0_store.add(
                content=insight,
                category="successful_strategy",
                metadata={"query": query[:100], "output": output[:100]},
            )
