"""Integration tests: full pipeline from input to output."""
import unittest
import sys
import os
import tempfile

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from engine import ConversationEngine


class TestIntegration(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        """Load the actual data files."""
        base_dir = os.path.join(os.path.dirname(__file__), '..')
        cls.lexicon_path = os.path.join(base_dir, 'data', 'lexicon.brain')
        cls.graph_path = os.path.join(base_dir, 'data', 'graph.brain')

        if not os.path.isfile(cls.lexicon_path) or not os.path.isfile(cls.graph_path):
            raise unittest.SkipTest("Data files not found")

        cls.engine = ConversationEngine(cls.lexicon_path, cls.graph_path, seed=42)

    def test_engine_loads(self):
        self.assertIsNotNone(self.engine)
        self.assertIsNotNone(self.engine.graph)
        self.assertIsNotNone(self.engine.lexicon)

    def test_startup_summary(self):
        summary = self.engine.startup_summary()
        self.assertIn('Neuron Conversation Engine', summary)
        self.assertIn('nodes', summary)

    def test_process_greeting(self):
        response, trace, elapsed = self.engine.process_input("hello")
        self.assertIsInstance(response, str)
        self.assertTrue(len(response) > 0)
        self.assertGreater(elapsed, 0)

    def test_process_minnesota(self):
        response, trace, elapsed = self.engine.process_input("tell me about minnesota")
        self.assertIsInstance(response, str)
        self.assertTrue(len(response) > 0)

    def test_process_weather(self):
        response, trace, elapsed = self.engine.process_input("how cold is winter?")
        self.assertIsInstance(response, str)
        self.assertTrue(len(response) > 0)

    def test_process_sports(self):
        response, trace, elapsed = self.engine.process_input("tell me about the vikings")
        self.assertIsInstance(response, str)
        self.assertTrue(len(response) > 0)

    def test_process_lakes(self):
        response, trace, elapsed = self.engine.process_input("what about the lakes?")
        self.assertIsInstance(response, str)
        self.assertTrue(len(response) > 0)

    def test_trace_has_content(self):
        response, trace, elapsed = self.engine.process_input("minnesota culture")
        compact = trace.format_compact()
        self.assertIn('THOUGHT TRACE', compact)

    def test_trace_full(self):
        response, trace, elapsed = self.engine.process_input("snow and cold")
        full = trace.format_full()
        self.assertIn('FULL THOUGHT TRACE', full)

    def test_deterministic(self):
        """Two engines with same seed should produce same output."""
        engine1 = ConversationEngine(self.lexicon_path, self.graph_path, seed=99)
        engine2 = ConversationEngine(self.lexicon_path, self.graph_path, seed=99)

        r1, _, _ = engine1.process_input("hello minnesota")
        r2, _, _ = engine2.process_input("hello minnesota")
        self.assertEqual(r1, r2)

    def test_memory_across_turns(self):
        engine = ConversationEngine(self.lexicon_path, self.graph_path, seed=55)
        engine.process_input("hello")
        engine.process_input("tell me about lakes")
        self.assertEqual(engine.turn_count, 2)
        self.assertEqual(len(engine.memory.episodes), 2)

    def test_show_brain(self):
        info = self.engine.show_brain()
        self.assertIn('concept', info)
        self.assertIn('goal', info)

    def test_modulator_update_on_question(self):
        engine = ConversationEngine(self.lexicon_path, self.graph_path, seed=77)
        initial_curiosity = engine.modulators['curiosity']
        engine.process_input("what is minnesota?")
        self.assertGreater(engine.modulators['curiosity'], initial_curiosity)

    def test_seed_change(self):
        engine = ConversationEngine(self.lexicon_path, self.graph_path, seed=42)
        engine.set_seed(123)
        self.assertEqual(engine.seed, 123)

    def test_graph_has_minimum_nodes(self):
        self.assertGreaterEqual(len(self.engine.graph.nodes), 128)

    def test_graph_has_minimum_edges(self):
        self.assertGreaterEqual(len(self.engine.graph.edges), 400)

    def test_lexicon_has_minimum_entries(self):
        total = len(self.engine.lexicon.words) + len(self.engine.lexicon.phrases)
        self.assertGreaterEqual(total, 96)

    def test_phrase_input(self):
        response, trace, elapsed = self.engine.process_input("I love the twin cities and state fair")
        self.assertIsInstance(response, str)
        self.assertTrue(len(response) > 0)

    def test_empty_input_concepts(self):
        engine = ConversationEngine(self.lexicon_path, self.graph_path, seed=42)
        response, trace, elapsed = engine.process_input("the a is and")
        self.assertIsInstance(response, str)


if __name__ == '__main__':
    unittest.main()
