"""Tests for memory system."""
import unittest
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from engine.memory.memory import Memory, Episode


class TestMemory(unittest.TestCase):
    def test_store_turn(self):
        mem = Memory()
        mem.store_turn("hello", "hi there", ["c_greeting"], "goal_greet")
        self.assertEqual(mem.turn_counter, 1)
        self.assertEqual(len(mem.short_term), 1)
        self.assertEqual(len(mem.episodes), 1)

    def test_short_term_capacity(self):
        mem = Memory(stm_capacity=3)
        for i in range(5):
            mem.store_turn(f"user{i}", f"sys{i}", [f"c_{i}"], "goal_inform")
        self.assertEqual(len(mem.short_term), 3)
        # Oldest should be dropped
        self.assertEqual(mem.short_term[0][0], "user2")

    def test_episodic_capacity(self):
        mem = Memory(episodic_capacity=3)
        for i in range(5):
            mem.store_turn(f"user{i}", f"sys{i}", [f"c_{i}"], "goal_inform")
        self.assertEqual(len(mem.episodes), 3)

    def test_retrieve_relevant(self):
        mem = Memory()
        mem.store_turn("lakes", "water", ["c_lake", "c_water"], "goal_inform")
        mem.store_turn("sports", "football", ["c_sports", "c_football"], "goal_inform")
        mem.store_turn("cold", "snow", ["c_cold", "c_snow"], "goal_inform")

        retrieved = mem.retrieve_relevant(["c_lake", "c_water"])
        self.assertTrue(len(retrieved) > 0)
        # Should find the lakes episode
        lake_found = any("c_lake" in ep.concepts for ep in retrieved)
        self.assertTrue(lake_found)

    def test_retrieve_no_concepts(self):
        mem = Memory()
        mem.store_turn("hello", "hi", ["c_greeting"], "goal_greet")
        retrieved = mem.retrieve_relevant([])
        self.assertEqual(len(retrieved), 0)

    def test_retrieve_empty_memory(self):
        mem = Memory()
        retrieved = mem.retrieve_relevant(["c_lake"])
        self.assertEqual(len(retrieved), 0)

    def test_get_recent_concepts(self):
        mem = Memory()
        mem.store_turn("hello", "hi", ["c_greeting"], "goal_greet")
        mem.store_turn("lakes", "water", ["c_lake"], "goal_inform")
        concepts = mem.get_recent_concepts()
        self.assertIn("c_greeting", concepts)
        self.assertIn("c_lake", concepts)

    def test_get_memory_boost(self):
        mem = Memory()
        mem.store_turn("lakes", "water", ["c_lake", "c_water"], "goal_inform")
        boosts = mem.get_memory_boost(["c_lake"])
        self.assertTrue(len(boosts) > 0)
        # All boosts should be clamped
        for v in boosts.values():
            self.assertLessEqual(v, 0.4)

    def test_memory_boost_empty(self):
        mem = Memory()
        boosts = mem.get_memory_boost(["c_lake"])
        self.assertEqual(len(boosts), 0)

    def test_episode_structure(self):
        ep = Episode(1, "hello", "hi there", ["c_greeting"], "goal_greet")
        self.assertEqual(ep.turn_id, 1)
        self.assertEqual(ep.user_text, "hello")
        self.assertEqual(ep.system_text, "hi there")
        self.assertEqual(ep.concepts, ["c_greeting"])
        self.assertEqual(ep.goal, "goal_greet")

    def test_multiple_retrieval_boosts(self):
        mem = Memory()
        for i in range(5):
            mem.store_turn(f"lakes{i}", f"water{i}", ["c_lake", "c_water"], "goal_inform")
        boosts = mem.get_memory_boost(["c_lake"])
        # Should not exceed 0.4
        for v in boosts.values():
            self.assertLessEqual(v, 0.4)


if __name__ == '__main__':
    unittest.main()
