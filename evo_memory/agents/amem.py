"""Amem Agent implementation.

Amem (Agentic Memory) extends the ReAct pipeline with a lightweight
agentic memory that caches recent observations and reflections.

It provides a minimal form of experience reuse without dedicated
search or update policies.
"""

from typing import Tuple, Optional, List, Dict, Any
from dataclasses import dataclass, field

from .base import BaseAgent, AgentState, AgentAction, ActionType
from ..memory import Memory, MemoryEntry, Retriever, ContextBuilder
from ..memory.context import SimpleContextBuilder
from ..llm import BaseLLM


@dataclass
class AmemCache:
    """Lightweight cache for Amem agent."""
    observations: List[str] = field(default_factory=list)
    reflections: List[str] = field(default_factory=list)
    max_size: int = 10

    def add_observation(self, obs: str) -> None:
        """Add observation to cache."""
        self.observations.append(obs)
        if len(self.observations) > self.max_size:
            self.observations.pop(0)

    def add_reflection(self, reflection: str) -> None:
        """Add reflection to cache."""
        self.reflections.append(reflection)
        if len(self.reflections) > self.max_size:
            self.reflections.pop(0)

    def get_context(self) -> str:
        """Get cached context as string."""
        parts = []
        if self.observations:
            parts.append("Recent Observations:\n" + "\n".join(self.observations[-5:]))
        if self.reflections:
            parts.append("Reflections:\n" + "\n".join(self.reflections[-3:]))
        return "\n\n".join(parts)

    def clear(self) -> None:
        """Clear cache."""
        self.observations.clear()
        self.reflections.clear()


class AmemAgent(BaseAgent):
    """
    Amem Agent: Agentic Memory.

    Extends ReAct with a lightweight cache for observations and reflections.
    This provides minimal experience reuse without full memory management.
    """

    def __init__(
        self,
        llm: BaseLLM,
        retriever: Retriever,
        context_builder: Optional[ContextBuilder] = None,
        memory: Optional[Memory] = None,
        cache_size: int = 10,
        enable_reflection: bool = True,
        **kwargs,
    ):
        """
        Initialize Amem agent.

        Args:
            llm: Base LLM for generation
            retriever: Retriever for memory search
            context_builder: Optional context builder
            memory: Optional memory (used for cross-task persistence)
            cache_size: Size of the observation/reflection cache
            enable_reflection: Whether to generate reflections
        """
        context_builder = context_builder or SimpleContextBuilder()

        super().__init__(
            llm=llm,
            retriever=retriever,
            context_builder=context_builder,
            memory=memory,
            **kwargs,
        )

        self.cache = AmemCache(max_size=cache_size)
        self.enable_reflection = enable_reflection

    def run_single_turn(
        self,
        task_id: str,
        query: str,
        **kwargs,
    ) -> Tuple[str, AgentState]:
        """Run Amem on a single-turn task."""
        self.total_tasks += 1

        # Initialize state
        state = AgentState(
            task_id=task_id,
            input_text=query,
            memory=self.memory,
        )

        # Get cached context
        cached_context = self.cache.get_context()

        # Search memory if available
        if self.memory and len(self.memory) > 0:
            state.retrieved = self.search(query)

        # Build context
        context_parts = []
        if cached_context:
            context_parts.append(cached_context)
        if state.retrieved:
            context_parts.append(self.synthesize(query, state.retrieved))
        context_parts.append(f"Question: {query}")

        context = "\n\n".join(context_parts)

        # Generate response
        response = self.llm.generate(prompt=context)
        output = self.extract_answer(response.content)

        # Add to cache
        self.cache.add_observation(f"Q: {query[:100]}... A: {output[:100]}...")

        # Generate reflection if enabled
        if self.enable_reflection:
            reflection = self._generate_reflection(query, output)
            if reflection:
                self.cache.add_reflection(reflection)

        # Update state
        state.final_output = output
        state.is_complete = True
        state.feedback = kwargs.get("feedback")
        state.is_successful = kwargs.get("is_correct", False)

        if state.is_successful:
            self.successful_tasks += 1

        # Evolve memory
        self.evolve(state)

        return output, state

    def run_multi_turn(
        self,
        task_id: str,
        goal: str,
        environment,
        **kwargs,
    ) -> Tuple[bool, float, AgentState]:
        """Run Amem on a multi-turn task."""
        self.total_tasks += 1

        # Clear cache for new task
        self.cache.clear()

        # Initialize state
        state = AgentState(
            task_id=task_id,
            input_text=goal,
            memory=self.memory,
        )

        # Get initial observation
        observation = environment.reset()
        state.observations.append(observation)
        self.cache.add_observation(observation)

        # Search memory
        if self.memory and len(self.memory) > 0:
            state.retrieved = self.search(goal)

        success = False
        progress = 0.0

        for step in range(self.max_steps):
            state.current_step = step

            # Build context with cache
            context = self._build_context(
                goal=goal,
                state=state,
                current_observation=observation,
                environment_info=kwargs.get("environment_info", ""),
            )

            # Generate action
            response = self.llm.generate(prompt=context)
            action = self.parse_response(response.content)
            state.action_history.append(action)

            # Execute action
            if action.action_type == ActionType.ACT:
                observation, reward, done, info = environment.step(action.content)
                state.observations.append(observation)
                self.cache.add_observation(f"Action: {action.content} -> {observation[:100]}...")

                if "progress" in info:
                    progress = info["progress"]

                if done:
                    success = info.get("success", False)
                    # Generate reflection on task completion
                    if self.enable_reflection:
                        reflection = self._generate_task_reflection(goal, success)
                        self.cache.add_reflection(reflection)
                    break

        # Final state
        state.is_complete = True
        state.is_successful = success
        state.feedback = "Success" if success else "Failure"
        state.final_output = state.action_history[-1].content if state.action_history else ""

        if success:
            self.successful_tasks += 1

        # Evolve memory
        self.evolve(state)

        return success, progress, state

    def _build_context(
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

        # Cached context (observations + reflections)
        cached = self.cache.get_context()
        if cached:
            parts.append(f"Memory Cache:\n{cached}")

        # Retrieved experiences
        if state.retrieved:
            exp_str = self.synthesize(goal, state.retrieved)
            parts.append(f"Similar Experiences:\n{exp_str}")

        # Current task
        parts.append(f"Goal: {goal}")
        parts.append(f"Current Observation: {current_observation}")
        parts.append("What action should you take? Respond with 'Action: <action>'")

        return "\n\n".join(parts)

    def _generate_reflection(self, query: str, answer: str) -> Optional[str]:
        """Generate a reflection on the completed task."""
        prompt = f"""Briefly reflect on this Q&A (one sentence):
Question: {query[:200]}
Answer: {answer[:200]}

What key insight or strategy was useful here?
Reflection:"""

        response = self.llm.generate(prompt=prompt, max_tokens=100)
        reflection = response.content.strip()

        if len(reflection) > 10:
            return reflection
        return None

    def _generate_task_reflection(self, goal: str, success: bool) -> str:
        """Generate reflection on completed task."""
        status = "succeeded" if success else "failed"
        return f"Task '{goal[:50]}...' {status}."

    def clear_cache(self) -> None:
        """Clear the agent's cache."""
        self.cache.clear()
