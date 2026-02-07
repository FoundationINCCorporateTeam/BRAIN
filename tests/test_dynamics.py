"""Tests for dynamics/spreading activation system."""
import unittest
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from engine.graph.node import Node
from engine.graph.edge import Edge
from engine.graph.brain import BrainGraph
from engine.cognition.dynamics import run_dynamics, DynamicsConfig, DynamicsResult


class TestDynamics(unittest.TestCase):
    def _make_simple_graph(self):
        """Create a simple graph: n1 -> n2 -> n3, with a goal node."""
        g = BrainGraph()
        g.add_node(Node('n1', 'concept', 'C1', baseline=0.0, threshold=0.3))
        g.add_node(Node('n2', 'concept', 'C2', baseline=0.0, threshold=0.3))
        g.add_node(Node('n3', 'concept', 'C3', baseline=0.0, threshold=0.3))
        g.add_node(Node('g1', 'goal', 'Goal1', baseline=0.1, threshold=0.2))
        g.add_edge(Edge('n1', 'n2', 'excitatory', 0.6))
        g.add_edge(Edge('n2', 'n3', 'excitatory', 0.5))
        g.add_edge(Edge('n1', 'g1', 'causal', 0.4))
        return g

    def test_activation_spreads(self):
        g = self._make_simple_graph()
        result = run_dynamics(g, {'n1': 0.8}, DynamicsConfig(steps=10))
        # n2 should receive activation from n1
        self.assertGreater(g.nodes['n2'].activation, 0.0)

    def test_no_initial_activation(self):
        g = self._make_simple_graph()
        result = run_dynamics(g, {}, DynamicsConfig(steps=5))
        # Only baseline activations; concepts have baseline=0
        self.assertAlmostEqual(g.nodes['n1'].activation, 0.0, places=1)

    def test_activation_clamped(self):
        g = self._make_simple_graph()
        result = run_dynamics(g, {'n1': 1.0}, DynamicsConfig(steps=20))
        for node in g.nodes.values():
            self.assertGreaterEqual(node.activation, 0.0)
            self.assertLessEqual(node.activation, 1.0)

    def test_inhibitory_edge(self):
        g = BrainGraph()
        g.add_node(Node('a', 'concept', 'A', baseline=0.0, threshold=0.2))
        g.add_node(Node('b', 'concept', 'B', baseline=0.0, threshold=0.2))
        g.add_node(Node('g1', 'goal', 'Goal1', baseline=0.1, threshold=0.2))
        g.add_edge(Edge('a', 'b', 'inhibitory', 0.8))
        result = run_dynamics(g, {'a': 0.9, 'b': 0.7}, DynamicsConfig(steps=10))
        # b should be suppressed
        self.assertLess(g.nodes['b'].activation, 0.7)

    def test_dynamics_result_structure(self):
        g = self._make_simple_graph()
        result = run_dynamics(g, {'n1': 0.8}, DynamicsConfig(steps=5))
        self.assertIsInstance(result, DynamicsResult)
        self.assertEqual(len(result.steps), 5)
        self.assertIn('n1', result.final_activations)

    def test_edge_contributions_tracked(self):
        g = self._make_simple_graph()
        result = run_dynamics(g, {'n1': 0.8}, DynamicsConfig(steps=10))
        self.assertTrue(len(result.top_contributing_edges) > 0)

    def test_step_records(self):
        g = self._make_simple_graph()
        result = run_dynamics(g, {'n1': 0.8}, DynamicsConfig(steps=5))
        for step in result.steps:
            self.assertIsNotNone(step.step_num)
            self.assertIsInstance(step.top_active, list)

    def test_modulators_affect_spread(self):
        g = self._make_simple_graph()
        # Add an associative edge
        g.add_edge(Edge('n1', 'n3', 'associative', 0.5))

        # High curiosity
        result_high = run_dynamics(g, {'n1': 0.8}, DynamicsConfig(steps=10),
                                    {'curiosity': 1.0, 'calm': 0.5, 'urgency': 0.3})
        act_high = g.nodes['n3'].activation

        # Low curiosity
        result_low = run_dynamics(g, {'n1': 0.8}, DynamicsConfig(steps=10),
                                   {'curiosity': 0.0, 'calm': 0.5, 'urgency': 0.3})
        act_low = g.nodes['n3'].activation

        # Higher curiosity should spread more via associative edges
        self.assertGreaterEqual(act_high, act_low)

    def test_competition_within_type(self):
        g = BrainGraph()
        g.add_node(Node('c1', 'concept', 'C1', baseline=0.0, threshold=0.2))
        g.add_node(Node('c2', 'concept', 'C2', baseline=0.0, threshold=0.2))
        g.add_node(Node('c3', 'concept', 'C3', baseline=0.0, threshold=0.2))
        g.add_node(Node('g1', 'goal', 'Goal1', baseline=0.1, threshold=0.2))
        result = run_dynamics(g, {'c1': 0.9, 'c2': 0.8, 'c3': 0.7},
                              DynamicsConfig(steps=10, competition_within_type=True))
        # c1 should maintain highest
        self.assertGreaterEqual(g.nodes['c1'].activation, g.nodes['c2'].activation)


if __name__ == '__main__':
    unittest.main()
