"""
BrainGraph: container for all nodes and edges.
Provides adjacency lookups and validation.
"""
from typing import Dict, List, Optional
from .node import Node
from .edge import Edge


class BrainGraph:
    def __init__(self):
        self.nodes: Dict[str, Node] = {}
        self.edges: List[Edge] = []
        self._outgoing: Dict[str, List[Edge]] = {}
        self._incoming: Dict[str, List[Edge]] = {}

    def add_node(self, node: Node):
        if node.id in self.nodes:
            raise ValueError(f"Duplicate node id: {node.id}")
        self.nodes[node.id] = node
        self._outgoing.setdefault(node.id, [])
        self._incoming.setdefault(node.id, [])

    def add_edge(self, edge: Edge):
        if edge.source_id not in self.nodes:
            raise ValueError(f"Edge source '{edge.source_id}' not found in graph")
        if edge.target_id not in self.nodes:
            raise ValueError(f"Edge target '{edge.target_id}' not found in graph")
        self.edges.append(edge)
        self._outgoing[edge.source_id].append(edge)
        self._incoming[edge.target_id].append(edge)

    def get_outgoing(self, node_id: str) -> List[Edge]:
        return self._outgoing.get(node_id, [])

    def get_incoming(self, node_id: str) -> List[Edge]:
        return self._incoming.get(node_id, [])

    def get_nodes_by_type(self, node_type: str) -> List[Node]:
        return [n for n in self.nodes.values() if n.type == node_type]

    def get_node(self, node_id: str) -> Optional[Node]:
        return self.nodes.get(node_id)

    def reset_activations(self):
        for node in self.nodes.values():
            node.reset()

    def reset_contributions(self):
        for edge in self.edges:
            edge.contribution = 0.0

    def validate(self) -> List[str]:
        errors = []
        for edge in self.edges:
            if edge.source_id not in self.nodes:
                errors.append(f"Edge references missing source: {edge.source_id}")
            if edge.target_id not in self.nodes:
                errors.append(f"Edge references missing target: {edge.target_id}")
        goals = self.get_nodes_by_type('goal')
        if not goals:
            errors.append("No goal nodes defined in graph")
        return errors

    def summary(self) -> str:
        return f"{len(self.nodes)} nodes, {len(self.edges)} edges"
