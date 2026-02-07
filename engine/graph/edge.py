"""
Edge types: excitatory, inhibitory, associative, causal
Each edge has: source_id, target_id, type, weight
"""


class Edge:
    __slots__ = ('source_id', 'target_id', 'type', 'weight', 'contribution')

    VALID_TYPES = ('excitatory', 'inhibitory', 'associative', 'causal')

    def __init__(self, source_id: str, target_id: str, edge_type: str, weight: float = 0.5):
        if edge_type not in self.VALID_TYPES:
            raise ValueError(f"Invalid edge type '{edge_type}', must be one of {self.VALID_TYPES}")
        if not -1.0 <= weight <= 1.0:
            raise ValueError(f"Weight {weight} out of range [-1, 1]")
        self.source_id = source_id
        self.target_id = target_id
        self.type = edge_type
        self.weight = weight
        self.contribution = 0.0

    def __repr__(self):
        return f"Edge({self.source_id}->{self.target_id}, {self.type}, w={self.weight:.3f})"
