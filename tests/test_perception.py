"""Tests for input perception system."""
import unittest
import sys
import os
import tempfile

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from engine.loaders.lexicon_loader import load_lexicon, Lexicon, LexiconEntry
from engine.perception.input_processor import InputProcessor, PerceptionResult


SAMPLE_LEXICON = """# Test lexicon
STOP|the
STOP|a
STOP|is
STOP|and
STOP|of
STOP|to
STOP|in
STOP|it

SYNONYM|hi|hello
SYNONYM|hey|hello
SYNONYM|howdy|hello

WORD|w_hello|hello|c_greeting,c_friendly|interjection
WORD|w_goodbye|goodbye|c_farewell|interjection
WORD|w_minnesota|minnesota|c_minnesota,c_state|noun
WORD|w_cold|cold|c_cold,c_winter|adj
WORD|w_lake|lake|c_lake,c_water|noun
WORD|w_snow|snow|c_snow,c_winter|noun
WORD|w_tell|tell|c_inform|verb
WORD|w_about|about|c_topic_marker|prep
WORD|w_what|what|c_question|pronoun
WORD|w_nice|nice|c_friendly,c_mn_nice|adj
WORD|w_vikings|vikings|c_vikings,c_sports|noun
WORD|w_fishing|fishing|c_fishing,c_recreation|noun

PHRASE|p_twin_cities|twin cities|c_twincities,c_city|noun
PHRASE|p_state_fair|state fair|c_statefair,c_culture|noun
PHRASE|p_lake_superior|lake superior|c_superior,c_lake|noun
PHRASE|p_mn_nice|minnesota nice|c_mn_nice,c_friendly|noun
"""


class TestLexiconLoader(unittest.TestCase):
    def setUp(self):
        self.tmpfile = tempfile.NamedTemporaryFile(mode='w', suffix='.brain',
                                                    delete=False)
        self.tmpfile.write(SAMPLE_LEXICON)
        self.tmpfile.close()
        self.lexicon = load_lexicon(self.tmpfile.name)

    def tearDown(self):
        os.unlink(self.tmpfile.name)

    def test_words_loaded(self):
        self.assertIn('hello', self.lexicon.words)
        self.assertIn('minnesota', self.lexicon.words)
        self.assertIn('cold', self.lexicon.words)

    def test_phrases_loaded(self):
        self.assertIn('twin cities', self.lexicon.phrases)
        self.assertIn('state fair', self.lexicon.phrases)

    def test_synonyms_loaded(self):
        self.assertEqual(self.lexicon.resolve('hi'), 'hello')
        self.assertEqual(self.lexicon.resolve('hey'), 'hello')
        self.assertEqual(self.lexicon.resolve('howdy'), 'hello')

    def test_stopwords_loaded(self):
        self.assertTrue(self.lexicon.is_stopword('the'))
        self.assertTrue(self.lexicon.is_stopword('a'))
        self.assertFalse(self.lexicon.is_stopword('minnesota'))

    def test_concept_to_words(self):
        words = self.lexicon.get_words_for_concept('c_greeting')
        self.assertIn('hello', words)

    def test_sorted_phrases(self):
        phrases = self.lexicon.get_sorted_phrases()
        self.assertTrue(len(phrases) > 0)
        # Longest first
        for i in range(len(phrases) - 1):
            self.assertGreaterEqual(len(phrases[i]), len(phrases[i+1]))

    def test_summary(self):
        s = self.lexicon.summary()
        self.assertIn('entries', s)

    def test_invalid_lexicon_raises(self):
        tmpfile = tempfile.NamedTemporaryFile(mode='w', suffix='.brain', delete=False)
        tmpfile.write("BADTYPE|data\n")
        tmpfile.close()
        try:
            with self.assertRaises(ValueError):
                load_lexicon(tmpfile.name)
        finally:
            os.unlink(tmpfile.name)


class TestInputProcessor(unittest.TestCase):
    def setUp(self):
        tmpfile = tempfile.NamedTemporaryFile(mode='w', suffix='.brain', delete=False)
        tmpfile.write(SAMPLE_LEXICON)
        tmpfile.close()
        self.tmppath = tmpfile.name
        self.lexicon = load_lexicon(self.tmppath)
        self.processor = InputProcessor(self.lexicon)

    def tearDown(self):
        os.unlink(self.tmppath)

    def test_basic_word_matching(self):
        result = self.processor.process("minnesota")
        self.assertIn('minnesota', result.tokens)
        self.assertTrue(len(result.matched_words) > 0)
        self.assertIn('c_minnesota', result.activated_concepts)

    def test_synonym_resolution(self):
        result = self.processor.process("hi")
        self.assertIn('hello', result.tokens)
        self.assertIn('hi', result.synonym_mappings)

    def test_stopword_removal(self):
        result = self.processor.process("the lake is cold")
        self.assertNotIn('the', result.tokens)
        self.assertNotIn('is', result.tokens)
        self.assertIn('lake', result.tokens)
        self.assertIn('cold', result.tokens)

    def test_phrase_detection(self):
        result = self.processor.process("I love the twin cities")
        self.assertTrue(len(result.matched_phrases) > 0)
        phrases_found = [p for p, _ in result.matched_phrases]
        self.assertIn('twin cities', phrases_found)

    def test_punctuation_removal(self):
        result = self.processor.process("Hello!")
        self.assertIn('hello', result.tokens)

    def test_case_normalization(self):
        result = self.processor.process("MINNESOTA")
        self.assertIn('minnesota', result.tokens)

    def test_concept_activation_clamped(self):
        result = self.processor.process("hello hello hello")
        for cid, val in result.activated_concepts.items():
            self.assertLessEqual(val, 1.0)

    def test_empty_input(self):
        result = self.processor.process("")
        self.assertEqual(len(result.tokens), 0)
        self.assertEqual(len(result.activated_concepts), 0)

    def test_all_stopwords(self):
        result = self.processor.process("the a is and of")
        self.assertEqual(len(result.tokens), 0)
        self.assertEqual(len(result.matched_words), 0)


if __name__ == '__main__':
    unittest.main()
