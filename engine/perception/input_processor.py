"""
Input perception system:
- Lowercase normalization
- Phrase detection (longest-match first)
- Tokenization
- Stopword removal
- Synonym normalization
- Token -> concept mapping
"""
import re
from typing import List, Tuple, Dict


class PerceptionResult:
    def __init__(self):
        self.raw_input: str = ""
        self.tokens: List[str] = []
        self.matched_phrases: List[Tuple[str, List[str]]] = []
        self.matched_words: List[Tuple[str, List[str]]] = []
        self.activated_concepts: Dict[str, float] = {}
        self.synonym_mappings: Dict[str, str] = {}
        self.removed_stopwords: List[str] = []


class InputProcessor:
    def __init__(self, lexicon):
        self.lexicon = lexicon
        self._sorted_phrases = lexicon.get_sorted_phrases()

    def process(self, raw_input: str) -> PerceptionResult:
        result = PerceptionResult()
        result.raw_input = raw_input

        text = raw_input.lower().strip()
        text = re.sub(r'[^\w\s]', '', text)

        remaining = text

        # Phase 1: Longest-match phrase detection
        for phrase in self._sorted_phrases:
            if phrase in remaining:
                entry = self.lexicon.lookup_phrase(phrase)
                if entry:
                    result.matched_phrases.append((phrase, entry.concept_ids))
                    for cid in entry.concept_ids:
                        result.activated_concepts[cid] = result.activated_concepts.get(cid, 0.0) + 0.8
                    remaining = remaining.replace(phrase, ' ')

        # Phase 2: Tokenize remaining text
        tokens = remaining.split()

        for token in tokens:
            canonical = self.lexicon.resolve(token)
            if canonical != token:
                result.synonym_mappings[token] = canonical
                token = canonical

            if self.lexicon.is_stopword(token):
                result.removed_stopwords.append(token)
                continue

            result.tokens.append(token)

            entry = self.lexicon.lookup_word(token)
            if entry:
                result.matched_words.append((token, entry.concept_ids))
                for cid in entry.concept_ids:
                    result.activated_concepts[cid] = result.activated_concepts.get(cid, 0.0) + 0.7

        # Clamp activations to max 1.0
        for cid in result.activated_concepts:
            result.activated_concepts[cid] = min(1.0, result.activated_concepts[cid])

        return result
