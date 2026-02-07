"""Tests for graph node and edge classes and BrainGraph container."""
import unittest
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from engine.graph.node import Node
from engine.graph.edge import Edge
from engine.graph.brain import BrainGraph


class TestNode(unittest.TestCase):
    def test_create_valid_node(self):
        node = Node('n1', 'concept', 'Test', baseline=0.1, decay=0.05, threshold=0.3)
        self.assertEqual(node.id, 'n1')
        self.assertEqual(node.type, 'concept')
        self.assertEqual(node.label, 'Test')
        self.assertAlmostEqual(node.activation, 0.1)
        self.assertAlmostEqual(node.baseline, 0.1)

    def test_invalid_type_raises(self):
        with self.assertRaises(ValueError):
            Node('n1', 'invalid_type', 'Test')

    def test_all_valid_types(self):
        for ntype in Node.VALID_TYPES:
            node = Node(f'n_{ntype}', ntype, ntype.title())
            self.assertEqual(node.type, ntype)

    def test_reset(self):
        node = Node('n1', 'concept', 'Test', baseline=0.2)
        node.activation = 0.9
        node.reset()
        self.assertAlmostEqual(node.activation, 0.2)

    def test_clamp_upper(self):
        node = Node('n1', 'concept', 'Test')
        node.activation = 1.5
        node.clamp()
        self.assertAlmostEqual(node.activation, 1.0)

    def test_clamp_lower(self):
        node = Node('n1', 'concept', 'Test')
        node.activation = -0.5
        node.clamp()
        self.assertAlmostEqual(node.activation, 0.0)

    def test_is_active(self):
        node = Node('n1', 'concept', 'Test', threshold=0.3)
        node.activation = 0.5
        self.assertTrue(node.is_active())
        node.activation = 0.1
        self.assertFalse(node.is_active())

    def test_repr(self):
        node = Node('n1', 'concept', 'Test')
        self.assertIn('n1', repr(node))
        self.assertIn('concept', repr(node))

    def test_metadata_default_empty(self):
        node = Node('n1', 'concept', 'Test')
        self.assertEqual(node.metadata, {})


class TestEdge(unittest.TestCase):
    def test_create_valid_edge(self):
        edge = Edge('a', 'b', 'excitatory', 0.5)
        self.assertEqual(edge.source_id, 'a')
        self.assertEqual(edge.target_id, 'b')
        self.assertEqual(edge.type, 'excitatory')
        self.assertAlmostEqual(edge.weight, 0.5)
        self.assertAlmostEqual(edge.contribution, 0.0)

    def test_invalid_type_raises(self):
        with self.assertRaises(ValueError):
            Edge('a', 'b', 'unknown', 0.5)

    def test_all_valid_types(self):
        for etype in Edge.VALID_TYPES:
            edge = Edge('a', 'b', etype, 0.5)
            self.assertEqual(edge.type, etype)

    def test_weight_out_of_range(self):
        with self.assertRaises(ValueError):
            Edge('a', 'b', 'excitatory', 1.5)
        with self.assertRaises(ValueError):
            Edge('a', 'b', 'excitatory', -1.5)

    def test_negative_weight_allowed(self):
        edge = Edge('a', 'b', 'inhibitory', -0.5)
        self.assertAlmostEqual(edge.weight, -0.5)

    def test_repr(self):
        edge = Edge('a', 'b', 'excitatory', 0.5)
        self.assertIn('a', repr(edge))
        self.assertIn('b', repr(edge))


class TestBrainGraph(unittest.TestCase):
    def _make_graph(self):
        g = BrainGraph()
        g.add_node(Node('n1', 'concept', 'C1'))
        g.add_node(Node('n2', 'concept', 'C2'))
        g.add_node(Node('g1', 'goal', 'Goal1'))
        g.add_edge(Edge('n1', 'n2', 'excitatory', 0.5))
        return g

    def test_add_node(self):
        g = BrainGraph()
        g.add_node(Node('n1', 'concept', 'C1'))
        self.assertIn('n1', g.nodes)

    def test_duplicate_node_raises(self):
        g = BrainGraph()
        g.add_node(Node('n1', 'concept', 'C1'))
        with self.assertRaises(ValueError):
            g.add_node(Node('n1', 'concept', 'C1 dup'))

    def test_add_edge(self):
        g = self._make_graph()
        self.assertEqual(len(g.edges), 1)

    def test_edge_missing_source_raises(self):
        g = BrainGraph()
        g.add_node(Node('n2', 'concept', 'C2'))
        with self.assertRaises(ValueError):
            g.add_edge(Edge('n1', 'n2', 'excitatory', 0.5))

    def test_edge_missing_target_raises(self):
        g = BrainGraph()
        g.add_node(Node('n1', 'concept', 'C1'))
        with self.assertRaises(ValueError):
            g.add_edge(Edge('n1', 'n2', 'excitatory', 0.5))

    def test_get_outgoing(self):
        g = self._make_graph()
        out = g.get_outgoing('n1')
        self.assertEqual(len(out), 1)
        self.assertEqual(out[0].target_id, 'n2')

    def test_get_incoming(self):
        g = self._make_graph()
        inc = g.get_incoming('n2')
        self.assertEqual(len(inc), 1)
        self.assertEqual(inc[0].source_id, 'n1')

    def test_get_nodes_by_type(self):
        g = self._make_graph()
        concepts = g.get_nodes_by_type('concept')
        self.assertEqual(len(concepts), 2)
        goals = g.get_nodes_by_type('goal')
        self.assertEqual(len(goals), 1)

    def test_get_node(self):
        g = self._make_graph()
        self.assertIsNotNone(g.get_node('n1'))
        self.assertIsNone(g.get_node('nonexistent'))

    def test_reset_activations(self):
        g = self._make_graph()
        g.nodes['n1'].activation = 0.9
        g.reset_activations()
        self.assertAlmostEqual(g.nodes['n1'].activation, g.nodes['n1'].baseline)

    def test_reset_contributions(self):
        g = self._make_graph()
        g.edges[0].contribution = 5.0
        g.reset_contributions()
        self.assertAlmostEqual(g.edges[0].contribution, 0.0)

    def test_validate_no_errors(self):
        g = self._make_graph()
        errors = g.validate()
        self.assertEqual(len(errors), 0)

    def test_validate_missing_goals(self):
        g = BrainGraph()
        g.add_node(Node('n1', 'concept', 'C1'))
        errors = g.validate()
        self.assertTrue(any('goal' in e.lower() for e in errors))

    def test_summary(self):
        g = self._make_graph()
        s = g.summary()
        self.assertIn('3 nodes', s)
        self.assertIn('1 edges', s)


if __name__ == '__main__':
    unittest.main()
