"""ReAct Agent implementation.

ReAct (Reasoning and Acting) synergizes reasoning and acting in language models.
This implementation follows Yao et al. (2022) as referenced in the paper.

ReAct generates interleaved reasoning traces and actions but does not
explicitly store or evolve information - memory is limited to immediate context.
"""

from typing import Tuple, Optional, List
import re

from .base import BaseAgent, AgentState, AgentAction, ActionType
from ..memory import Memory, Retriever, ContextBuilder
from ..memory.context import SimpleContextBuilder
from ..llm import BaseLLM


class ReActAgent(BaseAgent):
    """
    ReAct Agent: Reasoning and Acting.

    This agent interleaves reasoning (Thought) with action execution.
    Unlike ExpRAG/ReMem, it does not maintain persistent memory across tasks.
    """

    def __init__(
        self,
        llm: BaseLLM,
        retriever: Optional[Retriever] = None,
        context_builder: Optional[ContextBuilder] = None,
        max_steps: int = 50,
        max_iterations: int = 10,
        use_memory: bool = False,
        **kwargs,
    ):
        """
        Initialize ReAct agent.

        Args:
            llm: Base LLM for generation
            retriever: Optional retriever (only used if use_memory=True)
            context_builder: Optional context builder
            max_steps: Maximum steps for multi-turn tasks
            max_iterations: Maximum thought-action iterations per step
            use_memory: Whether to use memory (False for pure ReAct)
        """
        # ReAct typically doesn't use memory, but we support it for variants
        if retriever is None:
            from ..memory.retriever import RecencyRetriever
            retriever = RecencyRetriever()

        context_builder = context_builder or SimpleContextBuilder()

        super().__init__(
            llm=llm,
            retriever=retriever,
            context_builder=context_builder,
            memory=Memory() if use_memory else None,
            max_steps=max_steps,
            max_iterations=max_iterations,
            **kwargs,
        )

        self.use_memory = use_memory

    def run_single_turn(
        self,
        task_id: str,
        query: str,
        **kwargs,
    ) -> Tuple[str, AgentState]:
        """Run ReAct on a single-turn task."""
        self.total_tasks += 1

        # Initialize state
        state = AgentState(
            task_id=task_id,
            input_text=query,
            memory=self.memory or Memory(),
        )

        # Build prompt with ReAct format
        context = self._build_react_prompt(query)

        # Generate with ReAct reasoning
        response = self.llm.generate(
            prompt=context,
            system_prompt=self._get_system_prompt(),
        )

        # Parse the response for thought and action
        output = self._extract_final_answer(response.content)

        # Update state
        state.final_output = output
        state.is_complete = True
        state.feedback = kwargs.get("feedback")
        state.is_successful = kwargs.get("is_correct", False)

        if state.is_successful:
            self.successful_tasks += 1

        return output, state

    def run_multi_turn(
        self,
        task_id: str,
        goal: str,
        environment,
        **kwargs,
    ) -> Tuple[bool, float, AgentState]:
        """Run ReAct on a multi-turn task."""
        self.total_tasks += 1

        # Initialize state
        state = AgentState(
            task_id=task_id,
            input_text=goal,
            memory=self.memory or Memory(),
        )

        # Get initial observation
        observation = environment.reset()
        state.observations.append(observation)

        # Track thought-action trace
        trace = []
        success = False
        progress = 0.0

        for step in range(self.max_steps):
            state.current_step = step

            # Build ReAct prompt with history
            context = self._build_multi_turn_react_prompt(
                goal=goal,
                trace=trace,
                current_observation=observation,
                environment_info=kwargs.get("environment_info", ""),
            )

            # Generate thought and action
            response = self.llm.generate(prompt=context)

            # Parse response
            thought, action = self._parse_react_response(response.content)

            # Record in trace
            trace.append({
                "thought": thought,
                "action": action,
                "observation": observation,
            })

            state.action_history.append(AgentAction(
                action_type=ActionType.THINK,
                content=thought,
            ))
            state.action_history.append(AgentAction(
                action_type=ActionType.ACT,
                content=action,
            ))

            # Execute action
            observation, reward, done, info = environment.step(action)
            state.observations.append(observation)

            if "progress" in info:
                progress = info["progress"]

            if done:
                success = info.get("success", False)
                break

        # Final state
        state.is_complete = True
        state.is_successful = success
        state.feedback = "Success" if success else "Failure"
        state.final_output = trace[-1]["action"] if trace else ""

        if success:
            self.successful_tasks += 1

        return success, progress, state

    def _build_react_prompt(self, query: str) -> str:
        """Build ReAct-style prompt for single-turn tasks."""
        return f"""Solve the following problem step by step.

Question: {query}

Use the following format:
Thought: Think about what you need to do
Action: Perform any necessary computation or reasoning
... (repeat Thought/Action as needed)
Thought: I now have the answer
Final Answer: [your answer]

Begin!

Thought:"""

    def _build_multi_turn_react_prompt(
        self,
        goal: str,
        trace: List[dict],
        current_observation: str,
        environment_info: str = "",
    ) -> str:
        """Build ReAct prompt for multi-turn tasks."""
        parts = []

        if environment_info:
            parts.append(f"Environment: {environment_info}")

        parts.append(f"Goal: {goal}")
        parts.append("")

        # Add trace history
        for i, t in enumerate(trace[-5:]):  # Last 5 steps
            parts.append(f"Thought: {t['thought']}")
            parts.append(f"Action: {t['action']}")
            parts.append(f"Observation: {t['observation']}")
            parts.append("")

        parts.append(f"Observation: {current_observation}")
        parts.append("")
        parts.append("What should you do next?")
        parts.append("Thought:")

        return "\n".join(parts)

    def _parse_react_response(self, response: str) -> Tuple[str, str]:
        """Parse thought and action from ReAct response."""
        # Extract thought
        thought_match = re.search(
            r"Thought:\s*(.+?)(?=Action:|$)",
            response,
            re.IGNORECASE | re.DOTALL,
        )
        thought = thought_match.group(1).strip() if thought_match else response

        # Extract action
        action_match = re.search(
            r"Action:\s*(.+?)(?=Thought:|Observation:|$)",
            response,
            re.IGNORECASE | re.DOTALL,
        )
        action = action_match.group(1).strip() if action_match else "look around"

        return thought, action

    def _extract_final_answer(self, response: str) -> str:
        """Extract final answer from ReAct response."""
        # Try Final Answer pattern
        match = re.search(
            r"Final Answer:\s*(.+?)(?:\n|$)",
            response,
            re.IGNORECASE | re.DOTALL,
        )
        if match:
            return match.group(1).strip()

        # Try to extract from last Action
        action_matches = re.findall(
            r"Action:\s*(.+?)(?=Thought:|Observation:|$)",
            response,
            re.IGNORECASE | re.DOTALL,
        )
        if action_matches:
            return action_matches[-1].strip()

        return response.strip()

    def _get_system_prompt(self) -> str:
        """Get system prompt for ReAct."""
        return """You are a helpful assistant that solves problems step by step.
Use the Thought/Action format to reason through problems.
Always end with a Final Answer when you have solved the problem."""
