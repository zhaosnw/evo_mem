"""ReMem Agent implementation.

ReMem (Reasoning, Acting, Memory) synergizes reasoning, action, and memory
refinement within a single decision loop.

As described in the paper (Section 3.3):
- Think: Produces internal reasoning traces for task decomposition
- Act: Executes operations or outputs responses
- Refine: Performs meta-reasoning over memory (retrieve, prune, organize)

At each step t, given input x_t, memory state M_t, and reasoning traces o_t^{1:n-1},
the agent selects one of three operations: a_t^n ∈ {Think, Act, Refine}
"""

from typing import Tuple, Optional, List, Dict, Any
import re

from .base import BaseAgent, AgentState, AgentAction, ActionType
from ..memory import Memory, MemoryEntry, Retriever, ContextBuilder
from ..memory.retriever import RetrievalResult
from ..memory.context import StructuredContextBuilder
from ..llm import BaseLLM


class ReMemAgent(BaseAgent):
    """
    ReMem Agent: Reasoning, Acting, and Memory refinement.

    This agent implements the Think-Act-Refine loop described in the paper,
    allowing for:
    1. Internal reasoning (Think) before acting
    2. Action execution (Act) in the environment
    3. Memory refinement (Refine) including pruning and reorganization
    """

    def __init__(
        self,
        llm: BaseLLM,
        retriever: Retriever,
        context_builder: Optional[ContextBuilder] = None,
        memory: Optional[Memory] = None,
        top_k: int = 4,
        max_steps: int = 50,
        max_iterations: int = 10,
        enable_pruning: bool = True,
        pruning_threshold: float = 0.3,
        **kwargs,
    ):
        """
        Initialize ReMem agent.

        Args:
            llm: Base LLM for generation
            retriever: Embedding retriever
            context_builder: Optional context builder
            memory: Optional pre-initialized memory
            top_k: Number of experiences to retrieve
            max_steps: Maximum steps for multi-turn tasks
            max_iterations: Maximum Think/Refine iterations per step
            enable_pruning: Enable memory pruning during Refine
            pruning_threshold: Threshold for pruning irrelevant memories
        """
        context_builder = context_builder or StructuredContextBuilder()

        super().__init__(
            llm=llm,
            retriever=retriever,
            context_builder=context_builder,
            memory=memory,
            top_k=top_k,
            max_steps=max_steps,
            max_iterations=max_iterations,
            **kwargs,
        )

        self.enable_pruning = enable_pruning
        self.pruning_threshold = pruning_threshold

        # Track pruning statistics
        self.total_pruned = 0
        self.prune_operations = 0

    def run_single_turn(
        self,
        task_id: str,
        query: str,
        **kwargs,
    ) -> Tuple[str, AgentState]:
        """
        Run ReMem on a single-turn task.

        Uses Think-Refine-Act loop for enhanced reasoning.
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

        # Think-Refine-Act loop
        reasoning_trace = []
        output = None

        for iteration in range(self.max_iterations):
            # Build context with current state
            context = self._build_single_turn_context(
                query=query,
                retrieved=state.retrieved,
                reasoning_trace=reasoning_trace,
            )

            # Generate response
            response = self.llm.generate(
                prompt=context,
                system_prompt=self._get_single_turn_system_prompt(),
            )

            # Parse action
            action = self.parse_response(response.content)
            state.action_history.append(action)

            if action.action_type == ActionType.THINK:
                # Record thinking
                reasoning_trace.append(f"Think: {action.content}")

            elif action.action_type == ActionType.REFINE:
                # Handle memory refinement
                self._handle_refinement(action, state)
                reasoning_trace.append(f"Refine: {action.content}")

            elif action.action_type == ActionType.ACT:
                # Extract final answer
                output = self.extract_answer(action.content)
                break

        # If no explicit action, extract from last response
        if output is None:
            output = self.extract_answer(response.content)

        # Update state
        state.final_output = output
        state.is_complete = True

        # Get feedback
        state.feedback = kwargs.get("feedback")
        state.is_successful = kwargs.get("is_correct", False)

        if state.is_successful:
            self.successful_tasks += 1

        # Evolve memory
        # Add rationale to metadata
        if reasoning_trace:
            state.action_history[0].metadata["rationale"] = "\n".join(reasoning_trace)

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
        Run ReMem on a multi-turn task.

        Uses the full Think-Act-Refine loop for each step.
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

            # Inner Think-Refine loop before acting
            action = self._think_refine_act_step(
                goal=goal,
                state=state,
                environment_info=kwargs.get("environment_info", ""),
            )

            # Execute action
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

    def _think_refine_act_step(
        self,
        goal: str,
        state: AgentState,
        environment_info: str = "",
    ) -> AgentAction:
        """
        Execute one step of Think-Refine-Act loop.

        Returns the final action to execute.
        """
        reasoning_trace = []

        for iteration in range(self.max_iterations):
            # Build context
            context = self._build_multi_turn_context(
                goal=goal,
                state=state,
                reasoning_trace=reasoning_trace,
                environment_info=environment_info,
            )

            # Generate response
            response = self.llm.generate(prompt=context)

            # Parse action
            action = self.parse_response(response.content)
            state.action_history.append(action)

            if action.action_type == ActionType.THINK:
                reasoning_trace.append(f"Think: {action.content}")

            elif action.action_type == ActionType.REFINE:
                self._handle_refinement(action, state)
                reasoning_trace.append(f"Refine: {action.content}")

            elif action.action_type == ActionType.ACT:
                return action

        # If loop exhausted, return last action or generate one
        return AgentAction(
            action_type=ActionType.ACT,
            content="look around",  # Default action
        )

    def _handle_refinement(self, action: AgentAction, state: AgentState) -> None:
        """
        Handle memory refinement operations.

        Supports:
        - Pruning: Remove specified experience IDs
        - Reorganize: Update memory organization (future)
        """
        if action.metadata.get("type") == "prune":
            # Parse IDs to prune
            ids_to_prune = self._parse_prune_ids(action.content, state.retrieved)

            if ids_to_prune:
                # Remove from retrieved set
                state.retrieved = [
                    r for i, r in enumerate(state.retrieved)
                    if (i + 1) not in ids_to_prune
                ]
                self.prune_operations += 1
                self.total_pruned += len(ids_to_prune)

    def _parse_prune_ids(
        self,
        prune_str: str,
        retrieved: List[RetrievalResult],
    ) -> List[int]:
        """Parse prune IDs from string like '1,3' or '2-4'."""
        ids = []
        max_id = len(retrieved)

        # Handle comma-separated: "1,3,5"
        for part in prune_str.split(","):
            part = part.strip()

            # Handle range: "2-4"
            if "-" in part:
                try:
                    start, end = part.split("-")
                    for i in range(int(start), int(end) + 1):
                        if 1 <= i <= max_id:
                            ids.append(i)
                except ValueError:
                    pass
            else:
                # Single ID
                try:
                    idx = int(part)
                    if 1 <= idx <= max_id:
                        ids.append(idx)
                except ValueError:
                    pass

        return ids

    def _build_single_turn_context(
        self,
        query: str,
        retrieved: List[RetrievalResult],
        reasoning_trace: List[str],
    ) -> str:
        """Build context for single-turn tasks."""
        parts = []

        # Retrieved memories
        if retrieved:
            memory_parts = []
            for i, result in enumerate(retrieved):
                entry = result.entry
                mem_text = f"""[Memory {i + 1}]
Question: {entry.input_text[:300]}...
Answer: {entry.output_text[:300]}...
Result: {'Success' if entry.is_successful else 'Failure'}"""
                memory_parts.append(mem_text)

            parts.append("Retrieved Memories:\n" + "\n\n".join(memory_parts))

        # Reasoning trace
        if reasoning_trace:
            parts.append("Reasoning Trace:\n" + "\n".join(reasoning_trace))

        # Current task
        parts.append(f"Question: {query}")

        # Instructions
        parts.append("""Respond in one of these formats:
- Think: <your reasoning> - for internal reasoning
- Think-Prune: <IDs> - to remove unhelpful memories (e.g., "1,3")
- Final Answer: <answer> - your final answer""")

        return "\n\n".join(parts)

    def _build_multi_turn_context(
        self,
        goal: str,
        state: AgentState,
        reasoning_trace: List[str],
        environment_info: str = "",
    ) -> str:
        """Build context for multi-turn tasks with ReMem format."""
        parts = []

        # Environment instructions
        if environment_info:
            parts.append(f"""==================================================
ENVIRONMENT INSTRUCTIONS
==================================================
{environment_info}""")

        # Retrieved experiences
        if state.retrieved:
            exp_parts = []
            for i, result in enumerate(state.retrieved):
                entry = result.entry
                trajectory_str = ""
                if entry.trajectory:
                    traj = " -> ".join([
                        f"{s.get('action', '')}"
                        for s in entry.trajectory[:5]
                    ])
                    trajectory_str = f"\nTrajectory: {traj}"

                exp_parts.append(f"""[Experience #{i + 1}]
Goal: {entry.input_text}{trajectory_str}
Correctness: {'Success' if entry.is_successful else 'Failure'}""")

            parts.append(f"""==================================================
RELEVANT EXPERIENCE FROM SIMILAR TASKS
==================================================
{chr(10).join(exp_parts)}""")

        # Current task
        parts.append(f"""==================================================
YOUR CURRENT TASK
==================================================
Goal: {goal}
Help: type 'check valid actions' if action fails
Help: type 'inventory' to check items""")

        # Recent history
        if state.observations:
            history_lines = []
            for i, obs in enumerate(state.observations[-10:]):
                history_lines.append(f"Observation: {obs}")
                # Find corresponding action
                act_idx = i + len(state.observations) - 10
                actions = [a for a in state.action_history if a.action_type == ActionType.ACT]
                if act_idx < len(actions):
                    history_lines.append(f"Action: {actions[act_idx].content}")

            parts.append(f"""==================================================
RECENT HISTORY
==================================================
{chr(10).join(history_lines)}""")

        # Reasoning trace for current step
        if reasoning_trace:
            parts.append(f"""==================================================
CURRENT REASONING
==================================================
{chr(10).join(reasoning_trace)}""")

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

    def _get_single_turn_system_prompt(self) -> str:
        """Get system prompt for single-turn tasks."""
        return """You are a helpful assistant with access to LOCAL EXPERIENCE MEMORY.
You can use the Think-Refine-Act loop:
- Think: reason about the problem internally
- Think-Prune: remove unhelpful memories by specifying IDs
- Final Answer: provide your final answer

Use retrieved memories to inform your reasoning, but prune irrelevant ones."""

    def parse_response(self, response: str) -> AgentAction:
        """Parse LLM response into an agent action with ReMem-specific patterns."""
        response = response.strip()

        # Check for Think-Prune
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

        # Check for Final Answer
        answer_match = re.match(r"Final Answer:\s*(.+)", response, re.IGNORECASE | re.DOTALL)
        if answer_match:
            return AgentAction(
                action_type=ActionType.ACT,
                content=answer_match.group(1).strip(),
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

    def get_statistics(self) -> Dict[str, Any]:
        """Get agent statistics including pruning stats."""
        stats = super().get_statistics()
        stats["pruning"] = {
            "total_pruned": self.total_pruned,
            "prune_operations": self.prune_operations,
        }
        return stats
