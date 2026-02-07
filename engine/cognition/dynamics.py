"""
Dynamics loop: multi-step simulation of activation spreading.
Per step:
  a) decay toward baseline
  b) spread activation via edges
  c) inhibition / competition
  d) clamp to [0,1]
Tracks edge contributions for credit assignment.
"""
from typing import Dict, List, Tuple
from ..graph.brain import BrainGraph
from ..graph.node import Node


class DynamicsConfig:
    def __init__(self, steps: int = 20, inhibition_strength: float = 0.15,
                 competition_within_type: bool = True):
        self.steps = steps
        self.inhibition_strength = inhibition_strength
        self.competition_within_type = competition_within_type


class StepRecord:
    """Record of a single simulation step."""
    def __init__(self, step_num: int):
        self.step_num = step_num
        self.top_active: List[Tuple[str, float]] = []


class DynamicsResult:
    def __init__(self):
        self.steps: List[StepRecord] = []
        self.final_activations: Dict[str, float] = {}
        self.top_contributing_edges: List[Tuple[str, str, str, float]] = []


def run_dynamics(graph: BrainGraph, initial_activations: Dict[str, float],
                 config: DynamicsConfig = None,
                 modulators: Dict[str, float] = None) -> DynamicsResult:
    """Run the dynamics loop on the graph."""
    if config is None:
        config = DynamicsConfig()
    if modulators is None:
        modulators = {'curiosity': 0.5, 'calm': 0.5, 'urgency': 0.3}

    result = DynamicsResult()

    graph.reset_activations()
    graph.reset_contributions()

    # Inject initial activations
    for node_id, value in initial_activations.items():
        node = graph.get_node(node_id)
        if node:
            node.activation = min(1.0, node.activation + value)

    curiosity_mod = modulators.get('curiosity', 0.5)

    for step in range(config.steps):
        record = StepRecord(step)

        # a) Decay toward baseline
        for node in graph.nodes.values():
            node.activation += (node.baseline - node.activation) * node.decay

        # b) Spread activation
        deltas: Dict[str, float] = {}
        for edge in graph.edges:
            src_node = graph.get_node(edge.source_id)
            if src_node and src_node.is_active():
                spread = src_node.activation * edge.weight

                if edge.type == 'inhibitory':
                    spread = -abs(spread)
                elif edge.type == 'associative':
                    spread *= (0.5 + curiosity_mod * 0.5)
                elif edge.type == 'causal':
                    spread *= 0.8

                deltas[edge.target_id] = deltas.get(edge.target_id, 0.0) + spread
                edge.contribution += abs(spread)

        # Apply deltas
        for node_id, delta in deltas.items():
            node = graph.get_node(node_id)
            if node:
                node.activation += delta

        # c) Inhibition / competition within same type
        if config.competition_within_type:
            for node_type in Node.VALID_TYPES:
                type_nodes = graph.get_nodes_by_type(node_type)
                if len(type_nodes) <= 1:
                    continue
                active_nodes = [n for n in type_nodes if n.is_active()]
                if len(active_nodes) <= 1:
                    continue
                active_nodes.sort(key=lambda n: n.activation, reverse=True)
                for i, node in enumerate(active_nodes[1:], 1):
                    suppression = config.inhibition_strength * (i / len(active_nodes))
                    node.activation -= suppression

        # d) Clamp
        for node in graph.nodes.values():
            node.clamp()

        # Record top active nodes
        active = [(n.id, n.activation) for n in graph.nodes.values() if n.is_active()]
        active.sort(key=lambda x: x[1], reverse=True)
        record.top_active = active[:8]

        result.steps.append(record)

    # Collect final activations
    result.final_activations = {nid: n.activation for nid, n in graph.nodes.items()}

    # Top contributing edges
    sorted_edges = sorted(graph.edges, key=lambda e: e.contribution, reverse=True)
    result.top_contributing_edges = [
        (e.source_id, e.target_id, e.type, e.contribution)
        for e in sorted_edges[:10]
    ]

    return result
