"""LangMem Agent implementation.

LangMem implements LangChain-style memory with dynamic retrieval
and continual updates.

Based on LangChain contributors (2025) as referenced in the paper.
"""

from typing import Tuple, Optional, List, Dict, Any
from collections import deque

from .base import BaseAgent, AgentState, AgentAction, ActionType
from ..memory import Memory, MemoryEntry, Retriever, ContextBuilder
from ..memory.retriever import RetrievalResult
from ..memory.context import SimpleContextBuilder
from ..llm import BaseLLM


class ConversationBuffer:
    """Simple conversation buffer for LangMem."""

    def __init__(self, max_size: int = 20):
        self.buffer: deque = deque(maxlen=max_size)
        self.summary: str = ""

    def add(self, role: str, content: str) -> None:
        """Add a message to the buffer."""
        self.buffer.append({"role": role, "content": content})

    def get_history(self, last_n: int = None) -> List[Dict[str, str]]:
        """Get conversation history."""
        if last_n:
            return list(self.buffer)[-last_n:]
        return list(self.buffer)

    def get_history_text(self, last_n: int = None) -> str:
        """Get history as text."""
        history = self.get_history(last_n)
        return "\n".join([f"{m['role']}: {m['content']}" for m in history])

    def clear(self) -> None:
        """Clear buffer."""
        self.buffer.clear()
        self.summary = ""


class LangMemAgent(BaseAgent):
    """
    LangMem Agent: LangChain-style memory management.

    Features:
    - Conversation buffer memory
    - Summary memory for long conversations
    - Entity extraction and tracking
    """

    def __init__(
        self,
        llm: BaseLLM,
        retriever: Retriever,
        context_builder: Optional[ContextBuilder] = None,
        memory: Optional[Memory] = None,
        buffer_size: int = 20,
        enable_summary: bool = True,
        summary_threshold: int = 10,
        **kwargs,
    ):
        """
        Initialize LangMem agent.

        Args:
            llm: Base LLM for generation
            retriever: Retriever for semantic search
            context_builder: Optional context builder
            memory: Optional experience memory
            buffer_size: Size of conversation buffer
            enable_summary: Enable conversation summarization
            summary_threshold: Number of turns before summarizing
        """
        context_builder = context_builder or SimpleContextBuilder()

        super().__init__(
            llm=llm,
            retriever=retriever,
            context_builder=context_builder,
            memory=memory,
            **kwargs,
        )

        self.conversation_buffer = ConversationBuffer(max_size=buffer_size)
        self.enable_summary = enable_summary
        self.summary_threshold = summary_threshold
        self.entity_memory: Dict[str, str] = {}

    def run_single_turn(
        self,
        task_id: str,
        query: str,
        **kwargs,
    ) -> Tuple[str, AgentState]:
        """Run LangMem on a single-turn task."""
        self.total_tasks += 1

        state = AgentState(
            task_id=task_id,
            input_text=query,
            memory=self.memory,
        )

        # Add query to buffer
        self.conversation_buffer.add("user", query)

        # Search experience memory
        if self.memory and len(self.memory) > 0:
            state.retrieved = self.search(query)

        # Extract entities
        entities = self._extract_entities(query)

        # Build context
        context = self._build_context(query, state.retrieved, entities)

        # Generate response
        response = self.llm.generate(prompt=context)
        output = self.extract_answer(response.content)

        # Add response to buffer
        self.conversation_buffer.add("assistant", output)

        # Check if summarization needed
        if self.enable_summary and len(self.conversation_buffer.buffer) >= self.summary_threshold:
            self._summarize_conversation()

        # Update entity memory
        self._update_entities(query, output)

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
        """Run LangMem on a multi-turn task."""
        self.total_tasks += 1

        # Clear buffer for new task
        self.conversation_buffer.clear()

        state = AgentState(
            task_id=task_id,
            input_text=goal,
            memory=self.memory,
        )

        observation = environment.reset()
        state.observations.append(observation)
        self.conversation_buffer.add("environment", observation)

        if self.memory and len(self.memory) > 0:
            state.retrieved = self.search(goal)

        success = False
        progress = 0.0

        for step in range(self.max_steps):
            state.current_step = step

            context = self._build_multi_turn_context(
                goal=goal,
                state=state,
                current_observation=observation,
                environment_info=kwargs.get("environment_info", ""),
            )

            response = self.llm.generate(prompt=context)
            action = self.parse_response(response.content)
            state.action_history.append(action)

            self.conversation_buffer.add("assistant", action.content)

            if action.action_type == ActionType.ACT:
                observation, reward, done, info = environment.step(action.content)
                state.observations.append(observation)
                self.conversation_buffer.add("environment", observation)

                if "progress" in info:
                    progress = info["progress"]

                if done:
                    success = info.get("success", False)
                    break

        state.is_complete = True
        state.is_successful = success
        state.feedback = "Success" if success else "Failure"
        state.final_output = state.action_history[-1].content if state.action_history else ""

        if success:
            self.successful_tasks += 1

        self.evolve(state)
        return success, progress, state

    def _extract_entities(self, text: str) -> Dict[str, str]:
        """Extract entities from text (simplified implementation)."""
        # Simple entity extraction - in production, use NER
        entities = {}
        for entity, description in self.entity_memory.items():
            if entity.lower() in text.lower():
                entities[entity] = description
        return entities

    def _update_entities(self, query: str, response: str) -> None:
        """Update entity memory from conversation."""
        # Simple heuristic - extract capitalized words as potential entities
        import re
        words = re.findall(r'\b[A-Z][a-z]+\b', query + " " + response)
        for word in words:
            if word not in self.entity_memory and len(word) > 2:
                self.entity_memory[word] = f"Mentioned in conversation"

    def _summarize_conversation(self) -> None:
        """Summarize conversation when buffer is full."""
        history = self.conversation_buffer.get_history_text()

        prompt = f"""Summarize this conversation in 2-3 sentences:

{history}

Summary:"""

        response = self.llm.generate(prompt=prompt, max_tokens=150)
        self.conversation_buffer.summary = response.content.strip()

        # Clear older entries but keep recent ones
        recent = list(self.conversation_buffer.buffer)[-5:]
        self.conversation_buffer.buffer.clear()
        for msg in recent:
            self.conversation_buffer.buffer.append(msg)

    def _build_context(
        self,
        query: str,
        retrieved: List[RetrievalResult],
        entities: Dict[str, str],
    ) -> str:
        """Build context for single-turn tasks."""
        parts = []

        # Summary if available
        if self.conversation_buffer.summary:
            parts.append(f"Previous Context Summary:\n{self.conversation_buffer.summary}")

        # Recent conversation
        history = self.conversation_buffer.get_history_text(last_n=5)
        if history:
            parts.append(f"Recent Conversation:\n{history}")

        # Entity memory
        if entities:
            entity_str = "\n".join([f"- {k}: {v}" for k, v in entities.items()])
            parts.append(f"Known Entities:\n{entity_str}")

        # Retrieved experiences
        if retrieved:
            exp_str = "\n".join([f"- {r.entry.output_text[:100]}..." for r in retrieved[:3]])
            parts.append(f"Relevant Experiences:\n{exp_str}")

        parts.append(f"Question: {query}")
        parts.append("Answer:")

        return "\n\n".join(parts)

    def _build_multi_turn_context(
        self,
        goal: str,
        state: AgentState,
        current_observation: str,
        environment_info: str = "",
    ) -> str:
        """Build context for multi-turn tasks."""
        parts = []

        if environment_info:
            parts.append(f"Environment: {environment_info}")

        if self.conversation_buffer.summary:
            parts.append(f"Summary: {self.conversation_buffer.summary}")

        if state.retrieved:
            exp_parts = [f"- {r.entry.input_text[:60]}..." for r in state.retrieved[:2]]
            parts.append("Similar:\n" + "\n".join(exp_parts))

        parts.append(f"Goal: {goal}")

        # Recent buffer history
        history = self.conversation_buffer.get_history_text(last_n=6)
        if history:
            parts.append(f"Recent:\n{history}")

        parts.append(f"Current: {current_observation}")
        parts.append("Action:")

        return "\n\n".join(parts)

    def clear_buffer(self) -> None:
        """Clear conversation buffer."""
        self.conversation_buffer.clear()

    def clear_entities(self) -> None:
        """Clear entity memory."""
        self.entity_memory.clear()
