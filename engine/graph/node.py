"""
Neuron node types for the brain graph.
Types: concept, topic, emotion, goal, motor, lexeme
Each node has: id, type, label, activation, baseline, decay, threshold
"""


class Node:
    __slots__ = ('id', 'type', 'label', 'activation', 'baseline', 'decay', 'threshold', 'metadata')

    VALID_TYPES = ('concept', 'topic', 'emotion', 'goal', 'motor', 'lexeme')

    def __init__(self, node_id: str, node_type: str, label: str,
                 baseline: float = 0.0, decay: float = 0.05, threshold: float = 0.3):
        if node_type not in self.VALID_TYPES:
            raise ValueError(f"Invalid node type '{node_type}', must be one of {self.VALID_TYPES}")
        self.id = node_id
        self.type = node_type
        self.label = label
        self.activation = baseline
        self.baseline = baseline
        self.decay = decay
        self.threshold = threshold
        self.metadata = {}

    def reset(self):
        self.activation = self.baseline

    def clamp(self):
        self.activation = max(0.0, min(1.0, self.activation))

    def is_active(self) -> bool:
        return self.activation >= self.threshold

    def __repr__(self):
        return f"Node({self.id}, {self.type}, act={self.activation:.3f})"
