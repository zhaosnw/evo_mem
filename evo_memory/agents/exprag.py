"""ExpRAG Agent implementation.

ExpRAG (Experience Retrieval and Aggregation) is a simple baseline that:
1. Retrieves k similar experiences from memory
2. Uses in-context learning to condition on retrieved examples
3. Appends new experiences to memory after each task

As described in the paper (Section 3.2):
- Memory entry: m_i = S(x_i, ŷ_i, f_i) encodes structured experience
- Retrieval: R_t = Top-k_{m_i ∈ M_t} φ(x_t, m_i)
- Generation: ŷ_t = F(x_t, R_t)
- Update: M_{t+1} = M_t ∪ {(x_t, ŷ_t, f_t)}
"""

from typing import Tuple, Optional, List, Dict, Any
import hashlib

from .base import BaseAgent, AgentState, AgentAction, ActionType
from ..memory import Memory, MemoryEntry, Retriever, ContextBuilder
from ..memory.retriever import RetrievalResult, RecencyRetriever
from ..memory.context import SimpleContextBuilder
from ..llm import BaseLLM


class ExpRAGAgent(BaseAgent):
    """
    Experience Retrieval and Aggregation Agent.

    A simple yet effective baseline that retrieves past task experiences
    and uses them as in-context examples for the current task.
    """

    def __init__(
        self,
        llm: BaseLLM,
        retriever: Retriever,
        context_builder: Optional[ContextBuilder] = None,
        memory: Optional[Memory] = None,
        top_k: int = 4,
        max_steps: int = 50,
        include_trajectory: bool = True,
        include_feedback: bool = True,
        **kwargs,
    ):
        """
        Initialize ExpRAG agent.

        Args:
            llm: Base LLM for generation
            retriever: Embedding retriever for similarity search
            context_builder: Optional context builder (uses SimpleContextBuilder by default)
            memory: Optional pre-initialized memory
            top_k: Number of experiences to retrieve
            max_steps: Maximum steps for multi-turn tasks
            include_trajectory: Include action trajectories in context
            include_feedback: Include feedback in context
        """
        context_builder = context_builder or SimpleContextBuilder(
            include_trajectory=include_trajectory,
            include_feedback=include_feedback,
        )

        super().__init__(
            llm=llm,
            retriever=retriever,
            context_builder=context_builder,
            memory=memory,
            top_k=top_k,
            max_steps=max_steps,
            **kwargs,
        )

        self.include_trajectory = include_trajectory
        self.include_feedback = include_feedback

    def run_single_turn(
        self,
        task_id: str,
        query: str,
        **kwargs,
    ) -> Tuple[str, AgentState]:
        """
        Run ExpRAG on a single-turn task.

        Process:
        1. Search: Retrieve k similar experiences
        2. Synthesize: Build context with retrieved experiences
        3. Generate: Produce output using LLM
        4. Evolve: Store experience in memory
        """
        self.total_tasks += 1

        # Initialize state
        state = AgentState(
            task_id=task_id,
            input_text=query,
            memory=self.memory,
        )

        # Search: Retrieve relevant experiences
        state.retrieved = self.search(query)

        # Synthesize: Build context
        context = self.synthesize(query, state.retrieved)

        # Generate: Get LLM response
        response = self.llm.generate(
            prompt=context,
            system_prompt=kwargs.get("system_prompt"),
        )

        # Extract answer
        output = self.extract_answer(response.content)

        # Update state
        state.final_output = output
        state.is_complete = True
        state.action_history.append(AgentAction(
            action_type=ActionType.ACT,
            content=output,
        ))

        # Get feedback if provided
        state.feedback = kwargs.get("feedback")
        state.is_successful = kwargs.get("is_correct", False)

        if state.is_successful:
            self.successful_tasks += 1

        # Evolve: Store experience
        self.evolve(state)

        return output, state

    def run_multi_turn(
        self,
        task_id: str,
        goal: str,
        environment,
        **kwargs,
    ) -> Tuple[bool, float, AgentState]:
        """
        Run ExpRAG on a multi-turn task.

        Process:
        1. Initialize with goal
        2. For each step:
           a. Search memory for similar experiences
           b. Build context with current observation
           c. Generate action
           d. Execute action in environment
        3. Evolve memory with trajectory
        """
        self.total_tasks += 1

        # Initialize state
        state = AgentState(
            task_id=task_id,
            input_text=goal,
            memory=self.memory,
        )

        # Get initial observation
        observation = environment.reset()
        state.observations.append(observation)

        # Search for similar experiences
        state.retrieved = self.search(goal)

        # Multi-turn loop
        success = False
        progress = 0.0

        for step in range(self.max_steps):
            state.current_step = step

            # Build context with retrieved experiences and current history
            context = self._build_multi_turn_context(
                goal=goal,
                retrieved=state.retrieved,
                observations=state.observations,
                action_history=state.action_history,
                environment_info=kwargs.get("environment_info", ""),
            )

            # Generate action
            response = self.llm.generate(prompt=context)
            action = self.parse_response(response.content)

            # Record action
            state.action_history.append(action)

            # If it's an actual action, execute it
            if action.action_type == ActionType.ACT:
                observation, reward, done, info = environment.step(action.content)
                state.observations.append(observation)

                # Update progress
                if "progress" in info:
                    progress = info["progress"]

                if done:
                    success = info.get("success", False)
                    break

        # Final state update
        state.is_complete = True
        state.is_successful = success
        state.feedback = "Success" if success else "Failure"
        state.final_output = state.action_history[-1].content if state.action_history else ""

        if success:
            self.successful_tasks += 1

        # Evolve memory
        self.evolve(state)

        return success, progress, state

    def _build_multi_turn_context(
        self,
        goal: str,
        retrieved: List[RetrievalResult],
        observations: List[str],
        action_history: List[AgentAction],
        environment_info: str = "",
    ) -> str:
        """Build context for multi-turn tasks."""
        parts = []

        # Environment info
        if environment_info:
            parts.append(f"Environment:\n{environment_info}")

        # Retrieved experiences
        if retrieved:
            exp_parts = []
            for i, result in enumerate(retrieved):
                entry = result.entry
                exp_text = entry.to_text(
                    include_trajectory=self.include_trajectory,
                    include_feedback=self.include_feedback,
                )
                exp_parts.append(f"[Experience #{i + 1}]\n{exp_text}")

            parts.append("Similar Experiences:\n" + "\n\n".join(exp_parts))

        # Current task
        parts.append(f"Goal: {goal}")

        # History
        if observations:
            history_parts = []
            for i, obs in enumerate(observations):
                history_parts.append(f"Observation: {obs}")
                if i < len(action_history):
                    act = action_history[i]
                    history_parts.append(f"Action: {act.content}")

            parts.append("History:\n" + "\n".join(history_parts[-20:]))  # Last 10 turns

        # Instruction
        parts.append("What action should you take next? Respond with 'Action: <your action>'")

        return "\n\n".join(parts)


class ExpRecentAgent(ExpRAGAgent):
    """
    ExpRecent Agent - uses recency-based retrieval instead of similarity.

    This is a simpler baseline that retrieves the most recent experiences
    rather than the most similar ones.
    """

    def __init__(
        self,
        llm: BaseLLM,
        memory: Optional[Memory] = None,
        top_k: int = 4,
        **kwargs,
    ):
        """
        Initialize ExpRecent agent.

        Args:
            llm: Base LLM for generation
            memory: Optional pre-initialized memory
            top_k: Number of recent experiences to retrieve
        """
        # Use recency-based retriever
        retriever = RecencyRetriever()

        super().__init__(
            llm=llm,
            retriever=retriever,
            memory=memory,
            top_k=top_k,
            **kwargs,
        )
