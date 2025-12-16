"""SelfRAG Agent implementation.

SelfRAG (Self-Reflective Retrieval-Augmented Generation) integrates
dynamic retrieval and reflection to adaptively ground reasoning in prior contexts.

Based on Asai et al. (2024) as referenced in the paper.
"""

from typing import Tuple, Optional, List
import re

from .base import BaseAgent, AgentState, AgentAction, ActionType
from ..memory import Memory, MemoryEntry, Retriever, ContextBuilder
from ..memory.retriever import RetrievalResult
from ..memory.context import SimpleContextBuilder
from ..llm import BaseLLM


class SelfRAGAgent(BaseAgent):
    """
    SelfRAG Agent: Self-Reflective Retrieval-Augmented Generation.

    Features:
    1. Adaptive retrieval decision - decide when to retrieve
    2. Relevance assessment - evaluate retrieved content
    3. Self-critique - assess and refine own outputs
    """

    def __init__(
        self,
        llm: BaseLLM,
        retriever: Retriever,
        context_builder: Optional[ContextBuilder] = None,
        memory: Optional[Memory] = None,
        top_k: int = 4,
        retrieval_threshold: float = 0.5,
        critique_threshold: float = 0.7,
        **kwargs,
    ):
        """
        Initialize SelfRAG agent.

        Args:
            llm: Base LLM for generation
            retriever: Retriever for memory search
            context_builder: Optional context builder
            memory: Optional memory
            top_k: Number of items to retrieve
            retrieval_threshold: Threshold for retrieval decision
            critique_threshold: Threshold for self-critique
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

        self.retrieval_threshold = retrieval_threshold
        self.critique_threshold = critique_threshold

    def run_single_turn(
        self,
        task_id: str,
        query: str,
        **kwargs,
    ) -> Tuple[str, AgentState]:
        """Run SelfRAG on a single-turn task."""
        self.total_tasks += 1

        state = AgentState(
            task_id=task_id,
            input_text=query,
            memory=self.memory,
        )

        # Step 1: Decide whether to retrieve
        should_retrieve = self._should_retrieve(query)

        if should_retrieve and self.memory and len(self.memory) > 0:
            # Step 2: Retrieve and assess relevance
            state.retrieved = self.search(query)
            relevant_retrieved = self._assess_relevance(query, state.retrieved)
        else:
            relevant_retrieved = []

        # Step 3: Generate initial output
        context = self._build_context(query, relevant_retrieved)
        response = self.llm.generate(prompt=context)
        output = self.extract_answer(response.content)

        # Step 4: Self-critique and potentially refine
        critique_score, critique = self._self_critique(query, output)

        if critique_score < self.critique_threshold:
            # Refine output based on critique
            output = self._refine_output(query, output, critique, relevant_retrieved)

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
        """Run SelfRAG on a multi-turn task."""
        self.total_tasks += 1

        state = AgentState(
            task_id=task_id,
            input_text=goal,
            memory=self.memory,
        )

        observation = environment.reset()
        state.observations.append(observation)

        # Initial retrieval decision
        if self._should_retrieve(goal) and self.memory and len(self.memory) > 0:
            state.retrieved = self.search(goal)
            state.retrieved = self._assess_relevance(goal, state.retrieved)

        success = False
        progress = 0.0

        for step in range(self.max_steps):
            state.current_step = step

            # Build context with SelfRAG approach
            context = self._build_multi_turn_context(
                goal=goal,
                state=state,
                current_observation=observation,
                environment_info=kwargs.get("environment_info", ""),
            )

            # Generate action
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

        # Final state
        state.is_complete = True
        state.is_successful = success
        state.feedback = "Success" if success else "Failure"
        state.final_output = state.action_history[-1].content if state.action_history else ""

        if success:
            self.successful_tasks += 1

        self.evolve(state)
        return success, progress, state

    def _should_retrieve(self, query: str) -> bool:
        """Decide whether retrieval is needed for this query."""
        prompt = f"""Decide if external knowledge/examples would help answer this question.
Question: {query}

Answer YES or NO:"""

        response = self.llm.generate(prompt=prompt, max_tokens=10)
        return "yes" in response.content.lower()

    def _assess_relevance(
        self,
        query: str,
        retrieved: List[RetrievalResult],
    ) -> List[RetrievalResult]:
        """Assess and filter retrieved items by relevance."""
        relevant = []

        for result in retrieved:
            # Quick relevance check based on similarity score
            if result.score >= self.retrieval_threshold:
                relevant.append(result)

        return relevant

    def _self_critique(self, query: str, output: str) -> Tuple[float, str]:
        """Perform self-critique on the generated output."""
        prompt = f"""Critique this answer on a scale of 1-10.

Question: {query}
Answer: {output}

Provide:
Score: [1-10]
Critique: [brief critique]"""

        response = self.llm.generate(prompt=prompt, max_tokens=100)

        # Parse score
        score_match = re.search(r"Score:\s*(\d+)", response.content)
        score = int(score_match.group(1)) / 10 if score_match else 0.8

        # Parse critique
        critique_match = re.search(r"Critique:\s*(.+)", response.content, re.DOTALL)
        critique = critique_match.group(1).strip() if critique_match else ""

        return score, critique

    def _refine_output(
        self,
        query: str,
        output: str,
        critique: str,
        retrieved: List[RetrievalResult],
    ) -> str:
        """Refine output based on critique."""
        context_parts = [f"Question: {query}"]
        context_parts.append(f"Initial Answer: {output}")
        context_parts.append(f"Critique: {critique}")

        if retrieved:
            exp_parts = []
            for i, r in enumerate(retrieved[:2]):
                exp_parts.append(f"Reference {i+1}: {r.entry.output_text[:200]}...")
            context_parts.append("References:\n" + "\n".join(exp_parts))

        context_parts.append("Please provide an improved answer addressing the critique.")

        prompt = "\n\n".join(context_parts)
        response = self.llm.generate(prompt=prompt)

        return self.extract_answer(response.content)

    def _build_context(self, query: str, retrieved: List[RetrievalResult]) -> str:
        """Build context for single-turn tasks."""
        parts = []

        if retrieved:
            exp_parts = []
            for i, r in enumerate(retrieved):
                exp_parts.append(f"[Example {i+1}]\nQ: {r.entry.input_text[:200]}...\nA: {r.entry.output_text[:200]}...")
            parts.append("Relevant Examples:\n" + "\n\n".join(exp_parts))

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

        if state.retrieved:
            exp_parts = []
            for i, r in enumerate(state.retrieved[:3]):
                exp_parts.append(f"[Exp {i+1}] Goal: {r.entry.input_text[:100]}... Result: {'Success' if r.entry.is_successful else 'Failure'}")
            parts.append("Similar Tasks:\n" + "\n".join(exp_parts))

        parts.append(f"Goal: {goal}")

        # Recent history
        if state.observations:
            history = []
            for i, obs in enumerate(state.observations[-5:]):
                history.append(f"Obs: {obs}")
                if i < len(state.action_history):
                    history.append(f"Act: {state.action_history[i].content}")
            parts.append("History:\n" + "\n".join(history))

        parts.append(f"Current: {current_observation}")
        parts.append("Action:")

        return "\n\n".join(parts)
