"""
Goal selection: goals are nodes in the graph.
Multiple goals compete through activation.
Winning goal drives speech behavior.
"""
from typing import List, Optional, Tuple
from ..graph.brain import BrainGraph
from ..graph.node import Node


class GoalResult:
    def __init__(self):
        self.candidates: List[Tuple[str, float]] = []
        self.selected_goal: Optional[str] = None
        self.selected_activation: float = 0.0


def select_goal(graph: BrainGraph) -> GoalResult:
    """Select the highest-activation goal node."""
    result = GoalResult()

    goal_nodes = graph.get_nodes_by_type('goal')
    if not goal_nodes:
        return result

    candidates = [(g.id, g.activation) for g in goal_nodes]
    candidates.sort(key=lambda x: x[1], reverse=True)
    result.candidates = candidates

    if candidates and candidates[0][1] > 0.0:
        result.selected_goal = candidates[0][0]
        result.selected_activation = candidates[0][1]
    else:
        result.selected_goal = candidates[0][0]
        result.selected_activation = candidates[0][1]

    return result
