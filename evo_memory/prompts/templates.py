"""Prompt templates for Evo-Memory benchmark.

Contains templates for different agents and task types.
"""

from typing import Dict, List, Optional, Any
from string import Template


# =============================================================================
# Base Prompts
# =============================================================================

SYSTEM_PROMPT = """You are a helpful AI assistant. Provide clear and accurate responses."""

# =============================================================================
# ExpRAG Prompts
# =============================================================================

EXPRAG_SYSTEM = """You are an AI assistant that learns from past experiences.
Use the provided examples from similar tasks to guide your response.
Focus on patterns and approaches that led to success."""

EXPRAG_WITH_EXPERIENCES = """You are solving a task. Here are relevant examples from past experience:

$experiences

Now solve the following task:
$task

Think carefully about patterns from the examples above, then provide your answer."""

EXPRAG_NO_EXPERIENCES = """Solve the following task:
$task

Provide your answer directly."""

# =============================================================================
# ReMem Prompts (Think-Act-Refine)
# =============================================================================

REMEM_SYSTEM = """You are an AI agent that thinks, acts, and refines its memory.
You follow a Think-Act-Refine loop:
1. Think: Analyze the task and relevant memories
2. Act: Execute the task
3. Refine: Update your understanding based on results"""

REMEM_THINK_PROMPT = """Given the task and relevant memories, think through your approach.

Task: $task

Relevant Memories:
$memories

Think step by step:
1. What patterns do you see in the memories?
2. What approach would work best?
3. What pitfalls should you avoid?

Your analysis:"""

REMEM_ACT_PROMPT = """Based on your analysis, now execute the task.

Task: $task

Your Analysis:
$analysis

Provide your final answer:"""

REMEM_REFINE_PROMPT = """You have completed a task. Evaluate whether your memory needs refinement.

Task: $task
Your Answer: $answer
Result: $result

Current memories being evaluated:
$memories

Should any memories be:
1. Updated (if partially correct)?
2. Removed (if misleading)?
3. Merged (if redundant)?

Provide your refinement decision. Reply with either:
- KEEP: [memory_ids to keep]
- PRUNE: [memory_ids to prune]
- MERGE: [memory_ids to merge]
- UPDATE: [memory_id] [new content]

Or reply NONE if no changes needed."""

# =============================================================================
# ReAct Prompts
# =============================================================================

REACT_SYSTEM = """You are an AI agent that reasons and acts step by step.
Use the ReAct format:
Thought: Your reasoning
Action: The action to take
Observation: The result (will be provided)
... repeat until task is complete
Answer: Your final answer"""

REACT_PROMPT = """Task: $task

Previous interactions:
$history

Based on your observations, continue with:
Thought: [your reasoning]
Action: [your next action]

Or if ready to answer:
Answer: [your final answer]"""

# =============================================================================
# Self-RAG Prompts
# =============================================================================

SELFRAG_SYSTEM = """You are an AI assistant that self-reflects on its responses.
After generating a response, evaluate its quality and relevance."""

SELFRAG_GENERATE_PROMPT = """Task: $task

Relevant information:
$context

Generate a response to the task."""

SELFRAG_CRITIQUE_PROMPT = """Evaluate the following response:

Task: $task
Response: $response

Rate on a scale of 1-5:
- Relevance: How relevant is the response to the task?
- Accuracy: How accurate is the information?
- Completeness: Does it fully address the task?

Provide ratings and brief explanation:"""

SELFRAG_REFINE_PROMPT = """Based on the critique, improve your response.

Task: $task
Original Response: $response
Critique: $critique

Provide an improved response:"""

# =============================================================================
# Memory Management Prompts
# =============================================================================

MEMORY_SUMMARIZE_PROMPT = """Summarize the following task and its outcome for future reference:

Task: $task
Response: $response
Result: $result

Create a concise summary that captures:
1. What the task was about
2. The key approach used
3. Whether it was successful
4. Any important lessons learned

Summary:"""

MEMORY_MERGE_PROMPT = """The following memories are similar. Merge them into a single entry.

Memory 1: $memory1
Memory 2: $memory2

Create a merged memory that combines key information from both:"""

MEMORY_PRUNE_PROMPT = """Evaluate these memories and identify which are most valuable to keep.

Memories:
$memories

Criteria:
- Relevance to recent tasks
- Unique insights
- Success patterns

Rank from most to least valuable and explain:"""

# =============================================================================
# Multi-Turn Task Prompts
# =============================================================================

MULTITURN_SYSTEM = """You are an AI agent in an interactive environment.
You will receive observations and must choose actions to complete tasks.
Think carefully about each step and learn from the environment feedback."""

MULTITURN_ACTION_PROMPT = """Environment: $environment

Current observation: $observation

Task: $task

Available actions:
$actions

Previous experience with similar tasks:
$memories

What action should you take? Respond with just the action."""

MULTITURN_REFLECT_PROMPT = """You have completed a multi-turn task. Reflect on your performance.

Task: $task
Steps taken: $steps
Final result: $result

What did you learn? What would you do differently?

Reflection:"""

# =============================================================================
# Domain-Specific Prompts
# =============================================================================

MATH_PROMPT = """Solve the following math problem. Show your work step by step.

Problem: $task

$context

Solution:"""

SCIENCE_PROMPT = """Answer the following science question. Explain your reasoning.

Question: $task

$context

Answer:"""

API_PROMPT = """You need to use an API to accomplish a task.

Task: $task

Available APIs:
$apis

$context

Provide the API call in the format: API_NAME(param1=value1, param2=value2)

API Call:"""

# =============================================================================
# Template Helper Functions
# =============================================================================

def format_template(template: str, **kwargs) -> str:
    """
    Format a template string with given parameters.

    Args:
        template: Template string with $variable placeholders
        **kwargs: Variable values

    Returns:
        Formatted string
    """
    t = Template(template)
    return t.safe_substitute(**kwargs)


def format_experiences(memories: List[Dict[str, Any]], max_examples: int = 5) -> str:
    """
    Format memory entries as experience examples.

    Args:
        memories: List of memory entries
        max_examples: Maximum examples to include

    Returns:
        Formatted string of examples
    """
    if not memories:
        return "No relevant experiences found."

    examples = []
    for i, mem in enumerate(memories[:max_examples]):
        example = f"Example {i+1}:\n"
        example += f"  Task: {mem.get('input', mem.get('input_text', 'N/A'))}\n"
        example += f"  Approach: {mem.get('output', mem.get('output_text', 'N/A'))}\n"
        if mem.get('feedback'):
            example += f"  Result: {mem['feedback']}\n"
        examples.append(example)

    return "\n".join(examples)


def format_trajectory(trajectory: List[Dict[str, Any]], max_steps: int = 10) -> str:
    """
    Format action trajectory as history.

    Args:
        trajectory: List of step dictionaries
        max_steps: Maximum steps to include

    Returns:
        Formatted string
    """
    if not trajectory:
        return "No previous steps."

    steps = []
    for step in trajectory[-max_steps:]:
        step_str = f"Step {step.get('step', '?')}:\n"
        if 'observation' in step:
            step_str += f"  Observation: {step['observation']}\n"
        if 'action' in step:
            step_str += f"  Action: {step['action']}\n"
        if 'reward' in step:
            step_str += f"  Reward: {step['reward']}\n"
        steps.append(step_str)

    return "\n".join(steps)


def get_agent_prompts(agent_type: str) -> Dict[str, str]:
    """
    Get prompt templates for a specific agent type.

    Args:
        agent_type: Agent type name

    Returns:
        Dictionary of prompt templates
    """
    prompts = {
        "system": SYSTEM_PROMPT,
        "task": "$task",
    }

    if agent_type == "exprag" or agent_type == "exp_recent":
        prompts["system"] = EXPRAG_SYSTEM
        prompts["with_memories"] = EXPRAG_WITH_EXPERIENCES
        prompts["without_memories"] = EXPRAG_NO_EXPERIENCES

    elif agent_type == "remem":
        prompts["system"] = REMEM_SYSTEM
        prompts["think"] = REMEM_THINK_PROMPT
        prompts["act"] = REMEM_ACT_PROMPT
        prompts["refine"] = REMEM_REFINE_PROMPT

    elif agent_type == "react":
        prompts["system"] = REACT_SYSTEM
        prompts["react"] = REACT_PROMPT

    elif agent_type == "selfrag":
        prompts["system"] = SELFRAG_SYSTEM
        prompts["generate"] = SELFRAG_GENERATE_PROMPT
        prompts["critique"] = SELFRAG_CRITIQUE_PROMPT
        prompts["refine"] = SELFRAG_REFINE_PROMPT

    return prompts


def get_domain_prompt(domain: str) -> str:
    """
    Get domain-specific prompt template.

    Args:
        domain: Domain name (math, science, api, etc.)

    Returns:
        Prompt template
    """
    domain_prompts = {
        "math": MATH_PROMPT,
        "science": SCIENCE_PROMPT,
        "api": API_PROMPT,
        "general": "$task\n\n$context",
    }

    return domain_prompts.get(domain.lower(), domain_prompts["general"])
