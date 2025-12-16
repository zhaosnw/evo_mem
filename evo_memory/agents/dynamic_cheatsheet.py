"""Dynamic Cheatsheet Agent implementation.

Dynamic Cheatsheet emphasizes the reuse of procedural knowledge,
encoding "how-to" information rather than static facts.

Two variants:
- DC-Cu (Cumulative): Accumulates strategies over time
- DC-RS (Retrieval Synthesis): Retrieves and synthesizes relevant strategies

Based on Suzgun et al. (2025) as referenced in the paper.
"""

from typing import Tuple, Optional, List, Dict, Any
from enum import Enum

from .base import BaseAgent, AgentState, AgentAction, ActionType
from ..memory import Memory, MemoryEntry, Retriever, ContextBuilder
from ..memory.retriever import RetrievalResult
from ..memory.context import CheatsheetContextBuilder
from ..llm import BaseLLM


class CheatsheetMode(Enum):
    """Cheatsheet operation modes."""
    CUMULATIVE = "cumulative"  # DC-Cu
    SYNTHESIS = "synthesis"     # DC-RS


class DynamicCheatsheetAgent(BaseAgent):
    """
    Dynamic Cheatsheet Agent: Procedural knowledge reuse.

    Maintains a cheatsheet of strategies and procedures that can be:
    - Cumulative (DC-Cu): All strategies are kept and accumulated
    - Synthesis (DC-RS): Strategies are retrieved and synthesized
    """

    def __init__(
        self,
        llm: BaseLLM,
        retriever: Retriever,
        context_builder: Optional[ContextBuilder] = None,
        memory: Optional[Memory] = None,
        mode: CheatsheetMode = CheatsheetMode.CUMULATIVE,
        max_cheatsheet_size: int = 20,
        synthesis_top_k: int = 5,
        **kwargs,
    ):
        """
        Initialize Dynamic Cheatsheet agent.

        Args:
            llm: Base LLM for generation
            retriever: Retriever for strategy search
            context_builder: Optional context builder
            memory: Optional experience memory
            mode: Cheatsheet mode (cumulative or synthesis)
            max_cheatsheet_size: Maximum number of strategies to keep
            synthesis_top_k: Number of strategies to retrieve for synthesis
        """
        context_builder = context_builder or CheatsheetContextBuilder(
            mode=mode.value
        )

        super().__init__(
            llm=llm,
            retriever=retriever,
            context_builder=context_builder,
            memory=memory,
            **kwargs,
        )

        self.mode = mode
        self.max_cheatsheet_size = max_cheatsheet_size
        self.synthesis_top_k = synthesis_top_k

        # Cheatsheet storage
        self.cheatsheet: List[Dict[str, Any]] = []

    def run_single_turn(
        self,
        task_id: str,
        query: str,
        **kwargs,
    ) -> Tuple[str, AgentState]:
        """Run Dynamic Cheatsheet on a single-turn task."""
        self.total_tasks += 1

        state = AgentState(
            task_id=task_id,
            input_text=query,
            memory=self.memory,
        )

        # Get relevant strategies
        if self.mode == CheatsheetMode.SYNTHESIS:
            strategies = self._retrieve_strategies(query)
        else:
            strategies = self._get_cumulative_strategies()

        # Search experience memory
        if self.memory and len(self.memory) > 0:
            state.retrieved = self.search(query)

        # Build context with cheatsheet
        context = self._build_context(query, strategies, state.retrieved)

        # Generate response
        response = self.llm.generate(prompt=context)
        output = self.extract_answer(response.content)

        # Extract and store new strategy
        strategy = self._extract_strategy(query, output)
        if strategy:
            self._add_strategy(strategy, query)

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
        """Run Dynamic Cheatsheet on a multi-turn task."""
        self.total_tasks += 1

        state = AgentState(
            task_id=task_id,
            input_text=goal,
            memory=self.memory,
        )

        observation = environment.reset()
        state.observations.append(observation)

        # Get strategies
        if self.mode == CheatsheetMode.SYNTHESIS:
            strategies = self._retrieve_strategies(goal)
        else:
            strategies = self._get_cumulative_strategies()

        if self.memory and len(self.memory) > 0:
            state.retrieved = self.search(goal)

        success = False
        progress = 0.0

        for step in range(self.max_steps):
            state.current_step = step

            context = self._build_multi_turn_context(
                goal=goal,
                state=state,
                strategies=strategies,
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

        # Extract strategy from successful task
        if success:
            strategy = self._extract_task_strategy(goal, state.action_history)
            if strategy:
                self._add_strategy(strategy, goal)

        state.is_complete = True
        state.is_successful = success
        state.feedback = "Success" if success else "Failure"
        state.final_output = state.action_history[-1].content if state.action_history else ""

        if success:
            self.successful_tasks += 1

        self.evolve(state)
        return success, progress, state

    def _retrieve_strategies(self, query: str) -> List[str]:
        """Retrieve relevant strategies for DC-RS mode."""
        if not self.cheatsheet:
            return []

        # Simple keyword-based retrieval
        query_words = set(query.lower().split())
        scored = []

        for entry in self.cheatsheet:
            strategy_words = set(entry["strategy"].lower().split())
            context_words = set(entry.get("context", "").lower().split())

            overlap = len(query_words & (strategy_words | context_words))
            if overlap > 0:
                scored.append((entry["strategy"], overlap))

        scored.sort(key=lambda x: x[1], reverse=True)
        return [s for s, _ in scored[:self.synthesis_top_k]]

    def _get_cumulative_strategies(self) -> List[str]:
        """Get all strategies for DC-Cu mode."""
        return [entry["strategy"] for entry in self.cheatsheet[-self.max_cheatsheet_size:]]

    def _add_strategy(self, strategy: str, context: str) -> None:
        """Add a new strategy to the cheatsheet."""
        # Check for duplicates
        for entry in self.cheatsheet:
            if strategy.lower() == entry["strategy"].lower():
                return

        self.cheatsheet.append({
            "strategy": strategy,
            "context": context[:100],
        })

        # Limit size
        if len(self.cheatsheet) > self.max_cheatsheet_size * 2:
            self.cheatsheet = self.cheatsheet[-self.max_cheatsheet_size:]

    def _extract_strategy(self, query: str, output: str) -> Optional[str]:
        """Extract a reusable strategy from the task."""
        prompt = f"""Extract a general strategy or rule from this Q&A that could help solve similar problems.

Question: {query[:200]}
Answer: {output[:200]}

Strategy (one sentence, general and reusable):"""

        response = self.llm.generate(prompt=prompt, max_tokens=100)
        strategy = response.content.strip()

        if len(strategy) > 10 and len(strategy) < 200:
            return strategy
        return None

    def _extract_task_strategy(
        self,
        goal: str,
        action_history: List[AgentAction],
    ) -> Optional[str]:
        """Extract strategy from a multi-turn task."""
        actions = [a.content for a in action_history if a.action_type == ActionType.ACT][:5]
        actions_str = " -> ".join(actions)

        prompt = f"""Extract a general strategy from this successful task.

Goal: {goal[:100]}
Actions: {actions_str}

Strategy (one sentence):"""

        response = self.llm.generate(prompt=prompt, max_tokens=100)
        strategy = response.content.strip()

        if len(strategy) > 10:
            return strategy
        return None

    def _build_context(
        self,
        query: str,
        strategies: List[str],
        retrieved: List[RetrievalResult],
    ) -> str:
        """Build context with cheatsheet."""
        parts = []

        if strategies:
            strategy_str = "\n".join([f"- {s}" for s in strategies])
            parts.append(f"Strategy Cheatsheet:\n{strategy_str}")

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
        strategies: List[str],
        current_observation: str,
        environment_info: str = "",
    ) -> str:
        """Build context for multi-turn tasks."""
        parts = []

        if environment_info:
            parts.append(f"Environment: {environment_info}")

        if strategies:
            strategy_str = "\n".join([f"- {s}" for s in strategies[:5]])
            parts.append(f"Strategies:\n{strategy_str}")

        if state.retrieved:
            exp_parts = [f"- {r.entry.input_text[:60]}..." for r in state.retrieved[:2]]
            parts.append("Similar:\n" + "\n".join(exp_parts))

        parts.append(f"Goal: {goal}")

        if state.observations:
            history = [f"Obs: {obs}" for obs in state.observations[-3:]]
            parts.append("Recent:\n" + "\n".join(history))

        parts.append(f"Current: {current_observation}")
        parts.append("Action:")

        return "\n\n".join(parts)

    def clear_cheatsheet(self) -> None:
        """Clear the cheatsheet."""
        self.cheatsheet.clear()

    def get_cheatsheet(self) -> List[str]:
        """Get current cheatsheet strategies."""
        return [entry["strategy"] for entry in self.cheatsheet]
