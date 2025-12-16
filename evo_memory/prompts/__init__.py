"""Prompt templates for Evo-Memory benchmark."""

from .templates import (
    # Base prompts
    SYSTEM_PROMPT,
    # ExpRAG prompts
    EXPRAG_SYSTEM,
    EXPRAG_WITH_EXPERIENCES,
    EXPRAG_NO_EXPERIENCES,
    # ReMem prompts
    REMEM_SYSTEM,
    REMEM_THINK_PROMPT,
    REMEM_ACT_PROMPT,
    REMEM_REFINE_PROMPT,
    # ReAct prompts
    REACT_SYSTEM,
    REACT_PROMPT,
    # Self-RAG prompts
    SELFRAG_SYSTEM,
    SELFRAG_GENERATE_PROMPT,
    SELFRAG_CRITIQUE_PROMPT,
    SELFRAG_REFINE_PROMPT,
    # Memory prompts
    MEMORY_SUMMARIZE_PROMPT,
    MEMORY_MERGE_PROMPT,
    MEMORY_PRUNE_PROMPT,
    # Multi-turn prompts
    MULTITURN_SYSTEM,
    MULTITURN_ACTION_PROMPT,
    MULTITURN_REFLECT_PROMPT,
    # Domain prompts
    MATH_PROMPT,
    SCIENCE_PROMPT,
    API_PROMPT,
    # Helper functions
    format_template,
    format_experiences,
    format_trajectory,
    get_agent_prompts,
    get_domain_prompt,
)

__all__ = [
    # Base prompts
    "SYSTEM_PROMPT",
    # ExpRAG prompts
    "EXPRAG_SYSTEM",
    "EXPRAG_WITH_EXPERIENCES",
    "EXPRAG_NO_EXPERIENCES",
    # ReMem prompts
    "REMEM_SYSTEM",
    "REMEM_THINK_PROMPT",
    "REMEM_ACT_PROMPT",
    "REMEM_REFINE_PROMPT",
    # ReAct prompts
    "REACT_SYSTEM",
    "REACT_PROMPT",
    # Self-RAG prompts
    "SELFRAG_SYSTEM",
    "SELFRAG_GENERATE_PROMPT",
    "SELFRAG_CRITIQUE_PROMPT",
    "SELFRAG_REFINE_PROMPT",
    # Memory prompts
    "MEMORY_SUMMARIZE_PROMPT",
    "MEMORY_MERGE_PROMPT",
    "MEMORY_PRUNE_PROMPT",
    # Multi-turn prompts
    "MULTITURN_SYSTEM",
    "MULTITURN_ACTION_PROMPT",
    "MULTITURN_REFLECT_PROMPT",
    # Domain prompts
    "MATH_PROMPT",
    "SCIENCE_PROMPT",
    "API_PROMPT",
    # Helper functions
    "format_template",
    "format_experiences",
    "format_trajectory",
    "get_agent_prompts",
    "get_domain_prompt",
]
