"""
Memory system:
- Short-term memory: recent turns (user + system)
- Episodic memory: stores (concepts, user_text, system_output) per turn
- Retrieval boosts activation of recalled concepts
"""
from typing import List, Dict, Optional, Tuple


class Episode:
    __slots__ = ('turn_id', 'user_text', 'system_text', 'concepts', 'goal')

    def __init__(self, turn_id: int, user_text: str, system_text: str,
                 concepts: List[str], goal: str):
        self.turn_id = turn_id
        self.user_text = user_text
        self.system_text = system_text
        self.concepts = concepts
        self.goal = goal


class Memory:
    def __init__(self, stm_capacity: int = 5, episodic_capacity: int = 50):
        self.stm_capacity = stm_capacity
        self.episodic_capacity = episodic_capacity
        self.short_term: List[Tuple[str, str]] = []
        self.episodes: List[Episode] = []
        self.turn_counter: int = 0

    def store_turn(self, user_text: str, system_text: str,
                   concepts: List[str], goal: str):
        self.turn_counter += 1

        self.short_term.append((user_text, system_text))
        if len(self.short_term) > self.stm_capacity:
            self.short_term.pop(0)

        episode = Episode(self.turn_counter, user_text, system_text, concepts, goal)
        self.episodes.append(episode)
        if len(self.episodes) > self.episodic_capacity:
            self.episodes.pop(0)

    def retrieve_relevant(self, concepts: List[str], top_k: int = 3) -> List['Episode']:
        """Retrieve episodes most relevant to current concepts."""
        if not self.episodes or not concepts:
            return []

        concept_set = set(concepts)
        scored = []
        for ep in self.episodes:
            overlap = len(concept_set & set(ep.concepts))
            if overlap > 0:
                recency = ep.turn_id / max(1, self.turn_counter)
                score = overlap + recency * 0.3
                scored.append((score, ep))

        scored.sort(key=lambda x: x[0], reverse=True)
        return [ep for _, ep in scored[:top_k]]

    def get_recent_concepts(self) -> List[str]:
        """Get concepts from recent episodes."""
        concepts = []
        for ep in self.episodes[-3:]:
            concepts.extend(ep.concepts)
        return concepts

    def get_memory_boost(self, current_concepts: List[str]) -> Dict[str, float]:
        """Return activation boosts from memory retrieval."""
        boosts: Dict[str, float] = {}
        retrieved = self.retrieve_relevant(current_concepts)

        for ep in retrieved:
            for c in ep.concepts:
                boost = 0.15
                boosts[c] = boosts.get(c, 0.0) + boost

        for c in boosts:
            boosts[c] = min(0.4, boosts[c])

        return boosts
