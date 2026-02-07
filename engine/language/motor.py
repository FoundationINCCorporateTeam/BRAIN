"""
Language motor system:
Converts activated concepts into words using a motor-based process.
NO templates. NO canned sentences.

Process:
1. Identify active concepts and their associated lexemes
2. Score candidate words by concept activation, association strength,
   inhibition of recently used words, and POS sequencing constraints
3. Assemble word sequence with light syntax
"""
import random
from typing import List, Dict, Tuple, Optional


class WordCandidate:
    __slots__ = ('word', 'concept_id', 'activation', 'pos', 'score', 'reason')

    def __init__(self, word: str, concept_id: str, activation: float, pos: str):
        self.word = word
        self.concept_id = concept_id
        self.activation = activation
        self.pos = pos
        self.score = 0.0
        self.reason = ""


class MotorResult:
    def __init__(self):
        self.candidates_considered: List[WordCandidate] = []
        self.selected_words: List[WordCandidate] = []
        self.final_text: str = ""


# Light syntax transition rules: given current POS, what POS can follow
POS_TRANSITIONS = {
    'START': ['noun', 'adj', 'det', 'pronoun', 'interjection', 'verb', 'adverb'],
    'det': ['noun', 'adj'],
    'adj': ['noun', 'adj', 'conjunction'],
    'noun': ['verb', 'conjunction', 'prep', 'noun', 'adj', 'END'],
    'pronoun': ['verb', 'adverb'],
    'verb': ['noun', 'adj', 'det', 'adverb', 'prep', 'pronoun', 'END'],
    'adverb': ['verb', 'adj', 'adverb', 'END'],
    'prep': ['noun', 'det', 'adj', 'pronoun'],
    'conjunction': ['noun', 'det', 'adj', 'verb', 'pronoun'],
    'interjection': ['noun', 'det', 'pronoun', 'verb', 'END'],
}

GOAL_CONCEPT_BOOST = {
    'goal_inform': 0.3,
    'goal_greet': 0.4,
    'goal_describe': 0.3,
    'goal_farewell': 0.4,
    'goal_clarify': 0.2,
}


def generate_response(graph, lexicon, goal_id: str,
                       rng: random.Random,
                       recent_words: List[str] = None,
                       max_words: int = 15) -> MotorResult:
    """Generate a response from activated concepts."""
    if recent_words is None:
        recent_words = []

    result = MotorResult()
    recent_set = set(recent_words[-20:])

    # Step 1: Gather all active concepts and their activations
    # Prioritize 'concept' type nodes over 'topic' and 'emotion' since
    # concept nodes carry the lexical mappings
    active_concepts = []
    for node in graph.nodes.values():
        if node.is_active() and node.type in ('concept', 'topic', 'emotion'):
            # Give concept nodes a sort boost so they appear first
            priority = 1.0 if node.type == 'concept' else 0.5
            active_concepts.append((node.id, node.activation, priority))

    active_concepts.sort(key=lambda x: (x[2], x[1]), reverse=True)

    if not active_concepts:
        result.final_text = "i am processing"
        return result

    # Step 2: Build candidate words from active concepts
    all_candidates: List[WordCandidate] = []

    for concept_id, concept_act, _ in active_concepts[:25]:
        words = lexicon.get_words_for_concept(concept_id)
        for word_text in words:
            entry = lexicon.lookup_word(word_text) or lexicon.lookup_phrase(word_text)
            if entry:
                candidate = WordCandidate(word_text, concept_id, concept_act, entry.pos)

                base_score = concept_act * 0.6

                goal_node = graph.get_node(goal_id)
                if goal_node:
                    goal_boost = GOAL_CONCEPT_BOOST.get(goal_id, 0.1)
                    for edge in graph.get_outgoing(goal_id):
                        if edge.target_id == concept_id:
                            base_score += goal_boost * abs(edge.weight)
                            break

                if word_text in recent_set:
                    base_score *= 0.3

                candidate.score = base_score
                candidate.reason = f"concept={concept_id} act={concept_act:.2f} goal_match={goal_id}"
                all_candidates.append(candidate)

    # Also add motor/lexeme nodes that are active
    for node in graph.nodes.values():
        if node.is_active() and node.type in ('motor', 'lexeme'):
            words = lexicon.get_words_for_concept(node.id)
            for word_text in words:
                entry = lexicon.lookup_word(word_text) or lexicon.lookup_phrase(word_text)
                if entry:
                    candidate = WordCandidate(word_text, node.id, node.activation, entry.pos)
                    candidate.score = node.activation * 0.5
                    candidate.reason = f"motor/lexeme node={node.id}"
                    if word_text in recent_set:
                        candidate.score *= 0.3
                    all_candidates.append(candidate)

    result.candidates_considered = sorted(all_candidates, key=lambda c: c.score, reverse=True)

    # Step 3: Assemble word sequence using POS transitions
    current_pos = 'START'
    selected: List[WordCandidate] = []
    used_words = set()

    for _ in range(max_words):
        allowed_pos = POS_TRANSITIONS.get(current_pos, ['noun', 'verb', 'adj'])

        eligible = [c for c in all_candidates
                    if c.pos in allowed_pos and c.word not in used_words]

        if not eligible:
            eligible = [c for c in all_candidates if c.word not in used_words]
            if not eligible:
                break

        eligible.sort(key=lambda c: c.score, reverse=True)

        top_score = eligible[0].score
        top_tier = [c for c in eligible if c.score >= top_score * 0.85]

        if len(top_tier) > 1:
            chosen = top_tier[rng.randint(0, len(top_tier) - 1)]
        else:
            chosen = top_tier[0]

        selected.append(chosen)
        used_words.add(chosen.word)
        current_pos = chosen.pos

        if current_pos == 'END' or len(selected) >= max_words:
            break

        # Reduce score of same-concept candidates to encourage diversity
        for c in all_candidates:
            if c.concept_id == chosen.concept_id:
                c.score *= 0.5

    result.selected_words = selected
    result.final_text = ' '.join(w.word for w in selected)

    return result
