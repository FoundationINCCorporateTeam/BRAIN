"""
Lexicon loader: parses lexicon.brain file.
"""
from typing import Dict, List, Set, Tuple


class LexiconEntry:
    __slots__ = ('id', 'text', 'concept_ids', 'pos')

    def __init__(self, entry_id: str, text: str, concept_ids: List[str], pos: str):
        self.id = entry_id
        self.text = text
        self.concept_ids = concept_ids
        self.pos = pos


class Lexicon:
    def __init__(self):
        self.words: Dict[str, LexiconEntry] = {}
        self.phrases: Dict[str, LexiconEntry] = {}
        self.synonyms: Dict[str, str] = {}
        self.stopwords: Set[str] = set()
        self.concept_to_words: Dict[str, List[str]] = {}

    def add_word(self, entry: LexiconEntry):
        self.words[entry.text] = entry
        for cid in entry.concept_ids:
            self.concept_to_words.setdefault(cid, []).append(entry.text)

    def add_phrase(self, entry: LexiconEntry):
        self.phrases[entry.text] = entry
        for cid in entry.concept_ids:
            self.concept_to_words.setdefault(cid, []).append(entry.text)

    def add_synonym(self, synonym: str, canonical: str):
        self.synonyms[synonym] = canonical

    def add_stopword(self, word: str):
        self.stopwords.add(word)

    def resolve(self, word: str) -> str:
        """Resolve synonym to canonical form."""
        return self.synonyms.get(word, word)

    def is_stopword(self, word: str) -> bool:
        return word in self.stopwords

    def lookup_word(self, word: str):
        return self.words.get(word)

    def lookup_phrase(self, phrase: str):
        return self.phrases.get(phrase)

    def get_words_for_concept(self, concept_id: str) -> List[str]:
        return self.concept_to_words.get(concept_id, [])

    def get_sorted_phrases(self) -> List[str]:
        """Return phrases sorted by length (longest first) for matching."""
        return sorted(self.phrases.keys(), key=lambda p: len(p), reverse=True)

    def summary(self) -> str:
        total = len(self.words) + len(self.phrases)
        return f"{total} entries"


def load_lexicon(filepath: str) -> Lexicon:
    """Parse a lexicon.brain file and return a Lexicon object."""
    lexicon = Lexicon()
    errors = []

    with open(filepath, 'r') as f:
        for line_num, raw_line in enumerate(f, 1):
            line = raw_line.strip()
            if not line or line.startswith('#'):
                continue

            parts = line.split('|')
            record_type = parts[0].strip()

            try:
                if record_type == 'WORD':
                    if len(parts) < 5:
                        errors.append(f"Line {line_num}: WORD record needs 5 fields, got {len(parts)}")
                        continue
                    entry_id = parts[1].strip()
                    word = parts[2].strip().lower()
                    concept_ids = [c.strip() for c in parts[3].strip().split(',') if c.strip()]
                    pos = parts[4].strip()
                    lexicon.add_word(LexiconEntry(entry_id, word, concept_ids, pos))

                elif record_type == 'PHRASE':
                    if len(parts) < 5:
                        errors.append(f"Line {line_num}: PHRASE record needs 5 fields, got {len(parts)}")
                        continue
                    entry_id = parts[1].strip()
                    phrase = parts[2].strip().lower()
                    concept_ids = [c.strip() for c in parts[3].strip().split(',') if c.strip()]
                    pos = parts[4].strip()
                    lexicon.add_phrase(LexiconEntry(entry_id, phrase, concept_ids, pos))

                elif record_type == 'SYNONYM':
                    if len(parts) < 3:
                        errors.append(f"Line {line_num}: SYNONYM record needs 3 fields, got {len(parts)}")
                        continue
                    synonym = parts[1].strip().lower()
                    canonical = parts[2].strip().lower()
                    lexicon.add_synonym(synonym, canonical)

                elif record_type == 'STOP':
                    if len(parts) < 2:
                        errors.append(f"Line {line_num}: STOP record needs 2 fields, got {len(parts)}")
                        continue
                    lexicon.add_stopword(parts[1].strip().lower())

                else:
                    errors.append(f"Line {line_num}: Unknown record type '{record_type}'")

            except Exception as e:
                errors.append(f"Line {line_num}: Parse error: {e}")

    if errors:
        error_msg = "Lexicon validation errors:\n" + "\n".join(errors)
        raise ValueError(error_msg)

    return lexicon
