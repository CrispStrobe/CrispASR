"""Model-free transcript normalization and independent edit-alignment controls."""
import importlib.util
from pathlib import Path
import unittest

SOURCE = Path(__file__).resolve().parents[1]/'tools/kaggle/reuse-quality-cpu-20261112/word_error_rate.py'
spec = importlib.util.spec_from_file_location('word_error_rate',SOURCE)
wer = importlib.util.module_from_spec(spec)
spec.loader.exec_module(wer)


class WordErrorsTest(unittest.TestCase):
    def test_unicode_case_punctuation_and_hyphen_rules_are_explicit(self):
        self.assertEqual(wer.normalize_words('ＡＳＫ, Café! F-not F-what can’t under_score １２'),
                         ['ask','café','f','not','f','what','can','t','under','score','12'])
        self.assertEqual(wer.normalize_words('American Americans'),['american','americans'])

    def test_known_mixed_errors_and_alignment_reconstruction(self):
        result = wer.word_errors(['one','two','three','four'],['one','too','four','extra'])
        self.assertEqual(result['edits'],3)
        self.assertEqual(result['wer'],.75)
        self.assertEqual([x['reference'] for x in result['alignment'] if x['reference'] is not None],
                         ['one','two','three','four'])
        self.assertEqual([x['hypothesis'] for x in result['alignment'] if x['hypothesis'] is not None],
                         ['one','too','four','extra'])
        self.assertEqual(result['correct']+result['substitutions']+result['deletions'],4)
        self.assertEqual(result['correct']+result['substitutions']+result['insertions'],4)

    def test_each_edit_type_and_denominator(self):
        for ref,hyp,counts in [('a b','a x',(1,0,0)),('a b','a',(0,1,0)),('a','a x',(0,0,1))]:
            with self.subTest(ref=ref,hyp=hyp):
                result = wer.compare_transcripts(ref,hyp)
                self.assertEqual(tuple(result[k] for k in ('substitutions','deletions','insertions')),counts)
                self.assertEqual(result['reference_words'],len(ref.split()))
                self.assertEqual(result['wer'],1/len(ref.split()))

    def test_empty_reference_and_wer_greater_than_one(self):
        result = wer.compare_transcripts('', 'a b')
        self.assertIsNone(result['wer'])
        self.assertEqual(result['insertions'],2)
        self.assertEqual(wer.compare_transcripts('a','b c d')['wer'],3)
        self.assertEqual(wer.compare_transcripts('a b','')['deletions'],2)

    def test_ambiguous_alignment_tie_is_stable(self):
        first = wer.compare_transcripts('a b','b a')
        self.assertEqual(first,wer.compare_transcripts('a b','b a'))
        self.assertEqual(first['substitutions'],2)
        self.assertEqual(first['deletions'],0)
        self.assertEqual(first['insertions'],0)

    def test_multiple_speaker_references_do_not_share_state(self):
        references = {'speaker-a':'ONE two','speaker-b':'three four five'}
        hypotheses = {'speaker-a':'one too','speaker-b':'three four'}
        results = {s:wer.compare_transcripts(ref,hypotheses[s]) for s,ref in references.items()}
        self.assertEqual(results['speaker-a']['wer'],.5)
        self.assertEqual(results['speaker-b']['wer'],1/3)
        self.assertEqual(wer.compare_transcripts('one two','ONE, TWO!')['wer'],0)

    def test_malformed_inputs_fail_explicitly(self):
        with self.assertRaises(TypeError):wer.normalize_words(None)
        with self.assertRaises(TypeError):wer.word_errors('one',['one'])
        with self.assertRaises(ValueError):wer.word_errors([''],['one'])


if __name__ == '__main__':
    unittest.main()
