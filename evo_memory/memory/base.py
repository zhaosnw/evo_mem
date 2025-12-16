"""Base memory classes for Evo-Memory.

This module implements the core memory abstraction as described in the paper:
- Memory state M_t evolves with history
- Each memory entry captures (input, output, feedback)
- Supports various update strategies (append, compress, replace)
"""

from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional
from datetime import datetime
import json
import hashlib


@dataclass
class MemoryEntry:
    """
    A single memory entry that captures task experience.

    According to the paper: m_i = S(x_i, ŷ_i, f_i) encodes a structured
    experience text with template S.

    Attributes:
        task_id: Unique identifier for the task
        input_text: The original input/question (x_t)
        output_text: The model's output/answer (ŷ_t)
        feedback: Correctness signal or task feedback (f_t)
        trajectory: Optional action trajectory for multi-turn tasks
        metadata: Additional metadata (domain, skills, etc.)
        embedding: Cached embedding vector for retrieval
        timestamp: When this entry was created
        is_successful: Whether the task was completed successfully
    """
    task_id: str
    input_text: str
    output_text: str
    feedback: Optional[str] = None
    trajectory: Optional[List[Dict[str, str]]] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    embedding: Optional[List[float]] = None
    timestamp: datetime = field(default_factory=datetime.now)
    is_successful: bool = False

    def __post_init__(self):
        if not self.task_id:
            # Generate task_id from content hash
            content = f"{self.input_text}:{self.output_text}"
            self.task_id = hashlib.md5(content.encode()).hexdigest()[:12]

    def to_text(self, include_trajectory: bool = True, include_feedback: bool = True) -> str:
        """
        Convert memory entry to structured text format.

        This implements the template S(x_i, ŷ_i, f_i) from the paper.
        """
        parts = [f"Task: {self.input_text}"]

        if include_trajectory and self.trajectory:
            trajectory_str = "\n".join([
                f"  {step.get('action', '')}: {step.get('observation', '')}"
                for step in self.trajectory
            ])
            parts.append(f"Trajectory:\n{trajectory_str}")

        parts.append(f"Output: {self.output_text}")

        if include_feedback and self.feedback:
            parts.append(f"Feedback: {self.feedback}")

        if self.is_successful:
            parts.append("Result: Success")
        else:
            parts.append("Result: Failure")

        return "\n".join(parts)

    def to_dict(self) -> Dict[str, Any]:
        """Serialize memory entry to dictionary."""
        return {
            "task_id": self.task_id,
            "input_text": self.input_text,
            "output_text": self.output_text,
            "feedback": self.feedback,
            "trajectory": self.trajectory,
            "metadata": self.metadata,
            "timestamp": self.timestamp.isoformat(),
            "is_successful": self.is_successful,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "MemoryEntry":
        """Deserialize memory entry from dictionary."""
        data = data.copy()
        if "timestamp" in data and isinstance(data["timestamp"], str):
            data["timestamp"] = datetime.fromisoformat(data["timestamp"])
        # Remove embedding from dict as it's usually recomputed
        data.pop("embedding", None)
        return cls(**data)


class Memory:
    """
    Memory store that supports the search-synthesize-evolve loop.

    As described in the paper:
    - Search: R_t = R(M_t, x_t) - retrieve relevant entries
    - Synthesis: C̃_t = C(x_t, R_t) - build context
    - Evolve: M_{t+1} = U(M_t, m_t) - update memory

    This class manages the memory state M_t and supports various
    update strategies.
    """

    def __init__(
        self,
        max_size: int = 1000,
        enable_pruning: bool = True,
        pruning_threshold: float = 0.3,
        store_successful_only: bool = False,
    ):
        """
        Initialize memory store.

        Args:
            max_size: Maximum number of entries to store
            enable_pruning: Whether to enable memory pruning
            pruning_threshold: Threshold for pruning (based on relevance)
            store_successful_only: Only store successful experiences
        """
        self.max_size = max_size
        self.enable_pruning = enable_pruning
        self.pruning_threshold = pruning_threshold
        self.store_successful_only = store_successful_only

        self._entries: List[MemoryEntry] = []
        self._entry_map: Dict[str, int] = {}  # task_id -> index

        # Statistics
        self.total_added = 0
        self.total_pruned = 0

    def __len__(self) -> int:
        return len(self._entries)

    def __iter__(self):
        return iter(self._entries)

    def add(self, entry: MemoryEntry) -> bool:
        """
        Add a new memory entry (Evolve operation).

        Implements: M_{t+1} = U(M_t, m_t)

        Args:
            entry: Memory entry to add

        Returns:
            True if entry was added, False otherwise
        """
        # Check if we should store this entry
        if self.store_successful_only and not entry.is_successful:
            return False

        # Check for duplicates
        if entry.task_id in self._entry_map:
            # Update existing entry
            idx = self._entry_map[entry.task_id]
            self._entries[idx] = entry
            return True

        # Check capacity
        if len(self._entries) >= self.max_size:
            if self.enable_pruning:
                self._prune_oldest()
            else:
                return False

        self._entries.append(entry)
        self._entry_map[entry.task_id] = len(self._entries) - 1
        self.total_added += 1
        return True

    def get(self, task_id: str) -> Optional[MemoryEntry]:
        """Get a specific memory entry by task_id."""
        if task_id in self._entry_map:
            return self._entries[self._entry_map[task_id]]
        return None

    def get_all(self) -> List[MemoryEntry]:
        """Get all memory entries."""
        return list(self._entries)

    def get_recent(self, k: int = 5) -> List[MemoryEntry]:
        """Get k most recent entries."""
        return self._entries[-k:] if k < len(self._entries) else list(self._entries)

    def remove(self, task_id: str) -> bool:
        """Remove a specific memory entry."""
        if task_id not in self._entry_map:
            return False

        idx = self._entry_map[task_id]
        del self._entries[idx]
        del self._entry_map[task_id]

        # Update indices
        self._rebuild_index()
        self.total_pruned += 1
        return True

    def remove_by_ids(self, task_ids: List[str]) -> int:
        """Remove multiple entries by their task_ids."""
        removed = 0
        for task_id in task_ids:
            if self.remove(task_id):
                removed += 1
        return removed

    def prune_by_relevance(self, query_embedding: List[float], threshold: float) -> int:
        """
        Prune entries with low relevance to current context.

        This implements the memory refinement aspect of ReMem.
        """
        from .retriever import cosine_similarity

        to_remove = []
        for entry in self._entries:
            if entry.embedding is not None:
                similarity = cosine_similarity(query_embedding, entry.embedding)
                if similarity < threshold:
                    to_remove.append(entry.task_id)

        return self.remove_by_ids(to_remove)

    def _prune_oldest(self, count: int = 1) -> None:
        """Remove oldest entries to make room."""
        for _ in range(min(count, len(self._entries))):
            if self._entries:
                entry = self._entries.pop(0)
                self._entry_map.pop(entry.task_id, None)
                self.total_pruned += 1
        self._rebuild_index()

    def _rebuild_index(self) -> None:
        """Rebuild the task_id to index mapping."""
        self._entry_map = {
            entry.task_id: idx for idx, entry in enumerate(self._entries)
        }

    def clear(self) -> None:
        """Clear all memory entries."""
        self._entries.clear()
        self._entry_map.clear()

    def save(self, path: str) -> None:
        """Save memory to file."""
        data = {
            "entries": [e.to_dict() for e in self._entries],
            "stats": {
                "total_added": self.total_added,
                "total_pruned": self.total_pruned,
            }
        }
        with open(path, "w") as f:
            json.dump(data, f, indent=2)

    def load(self, path: str) -> None:
        """Load memory from file."""
        with open(path) as f:
            data = json.load(f)

        self._entries = [MemoryEntry.from_dict(e) for e in data["entries"]]
        self._rebuild_index()

        if "stats" in data:
            self.total_added = data["stats"].get("total_added", len(self._entries))
            self.total_pruned = data["stats"].get("total_pruned", 0)

    def get_statistics(self) -> Dict[str, Any]:
        """Get memory statistics."""
        successful = sum(1 for e in self._entries if e.is_successful)
        return {
            "total_entries": len(self._entries),
            "successful_entries": successful,
            "failed_entries": len(self._entries) - successful,
            "total_added": self.total_added,
            "total_pruned": self.total_pruned,
            "retention_rate": 1 - (self.total_pruned / max(self.total_added, 1)),
        }
