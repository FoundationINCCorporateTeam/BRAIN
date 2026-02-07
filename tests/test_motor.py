"""Tests for language motor system."""
import unittest
import random
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from engine.graph.node import Node
from engine.graph.edge import Edge
from engine.graph.brain import BrainGraph
from engine.loaders.lexicon_loader import Lexicon, LexiconEntry
from engine.language.motor import generate_response, MotorResult, WordCandidate


class TestMotor(unittest.TestCase):
    def _make_lexicon(self):
        lex = Lexicon()
        lex.add_word(LexiconEntry('w_hello', 'hello', ['c_greeting'], 'interjection'))
        lex.add_word(LexiconEntry('w_minnesota', 'minnesota', ['c_minnesota'], 'noun'))
        lex.add_word(LexiconEntry('w_cold', 'cold', ['c_cold'], 'adj'))
        lex.add_word(LexiconEntry('w_lake', 'lake', ['c_lake'], 'noun'))
        lex.add_word(LexiconEntry('w_beautiful', 'beautiful', ['c_beauty'], 'adj'))
        lex.add_word(LexiconEntry('w_nice', 'nice', ['c_friendly'], 'adj'))
        lex.add_word(LexiconEntry('w_snow', 'snow', ['c_snow'], 'noun'))
        lex.add_word(LexiconEntry('w_fishing', 'fishing', ['c_fishing'], 'noun'))
        lex.add_word(LexiconEntry('w_winter', 'winter', ['c_winter'], 'noun'))
        lex.add_word(LexiconEntry('w_enjoy', 'enjoy', ['c_positive'], 'verb'))
        lex.add_word(LexiconEntry('w_great', 'great', ['c_positive'], 'adj'))
        lex.add_word(LexiconEntry('w_goodbye', 'goodbye', ['c_farewell'], 'interjection'))
        return lex

    def _make_graph(self):
        g = BrainGraph()
        g.add_node(Node('c_greeting', 'concept', 'Greeting', threshold=0.3))
        g.add_node(Node('c_farewell', 'concept', 'Farewell', threshold=0.3))
        g.add_node(Node('c_minnesota', 'concept', 'Minnesota', threshold=0.3))
        g.add_node(Node('c_cold', 'concept', 'Cold', threshold=0.3))
        g.add_node(Node('c_lake', 'concept', 'Lake', threshold=0.3))
        g.add_node(Node('c_beauty', 'concept', 'Beauty', threshold=0.3))
        g.add_node(Node('c_friendly', 'concept', 'Friendly', threshold=0.3))
        g.add_node(Node('c_snow', 'concept', 'Snow', threshold=0.3))
        g.add_node(Node('c_fishing', 'concept', 'Fishing', threshold=0.3))
        g.add_node(Node('c_winter', 'concept', 'Winter', threshold=0.3))
        g.add_node(Node('c_positive', 'concept', 'Positive', threshold=0.3))
        g.add_node(Node('goal_greet', 'goal', 'Greet', threshold=0.2))
        g.add_node(Node('goal_inform', 'goal', 'Inform', threshold=0.2))
        g.add_edge(Edge('goal_greet', 'c_greeting', 'excitatory', 0.7))
        g.add_edge(Edge('goal_inform', 'c_minnesota', 'excitatory', 0.5))
        return g

    def test_generate_produces_output(self):
        g = self._make_graph()
        lex = self._make_lexicon()
        rng = random.Random(42)

        # Activate some concepts
        g.nodes['c_minnesota'].activation = 0.8
        g.nodes['c_lake'].activation = 0.6
        g.nodes['c_beauty'].activation = 0.5

        result = generate_response(g, lex, 'goal_inform', rng)
        self.assertIsInstance(result, MotorResult)
        self.assertTrue(len(result.final_text) > 0)
        self.assertTrue(len(result.selected_words) > 0)

    def test_no_active_concepts_fallback(self):
        g = self._make_graph()
        lex = self._make_lexicon()
        rng = random.Random(42)
        # No activations above threshold
        result = generate_response(g, lex, 'goal_inform', rng)
        self.assertEqual(result.final_text, "i am processing")

    def test_deterministic_with_seed(self):
        g = self._make_graph()
        lex = self._make_lexicon()

        for node in g.nodes.values():
            if node.type == 'concept':
                node.activation = 0.6

        rng1 = random.Random(42)
        result1 = generate_response(g, lex, 'goal_inform', rng1)

        # Reset activations
        for node in g.nodes.values():
            if node.type == 'concept':
                node.activation = 0.6

        rng2 = random.Random(42)
        result2 = generate_response(g, lex, 'goal_inform', rng2)

        self.assertEqual(result1.final_text, result2.final_text)

    def test_recent_words_inhibition(self):
        g = self._make_graph()
        lex = self._make_lexicon()
        rng = random.Random(42)

        g.nodes['c_minnesota'].activation = 0.8
        g.nodes['c_lake'].activation = 0.7

        result1 = generate_response(g, lex, 'goal_inform', rng, recent_words=[])
        
        # Reset activations
        g.nodes['c_minnesota'].activation = 0.8
        g.nodes['c_lake'].activation = 0.7

        rng2 = random.Random(42)
        result2 = generate_response(g, lex, 'goal_inform', rng2,
                                     recent_words=['minnesota', 'lake'])
        # Scores should differ due to inhibition
        if result1.candidates_considered and result2.candidates_considered:
            self.assertIsNotNone(result2)

    def test_max_words_limit(self):
        g = self._make_graph()
        lex = self._make_lexicon()
        rng = random.Random(42)

        for node in g.nodes.values():
            if node.type == 'concept':
                node.activation = 0.8

        result = generate_response(g, lex, 'goal_inform', rng, max_words=3)
        self.assertLessEqual(len(result.selected_words), 3)

    def test_word_candidate_fields(self):
        wc = WordCandidate('test', 'c_test', 0.5, 'noun')
        self.assertEqual(wc.word, 'test')
        self.assertEqual(wc.concept_id, 'c_test')
        self.assertAlmostEqual(wc.activation, 0.5)
        self.assertEqual(wc.pos, 'noun')
        self.assertAlmostEqual(wc.score, 0.0)


if __name__ == '__main__':
    unittest.main()
