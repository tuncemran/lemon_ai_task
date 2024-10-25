import unittest
from unittest.mock import patch, MagicMock
from asses_lexical_quality import assess_lexical_quality, process_dataset_in_batches
import textstat

class TestAssessLexicalQuality(unittest.TestCase):

    def test_assess_lexical_quality(self):
        text = "The quick brown fox jumps over the lazy dog."
        expected_output = {
            'spelling_accuracy': 1.0,  # Assuming no spelling errors
            'grammar_quality': 1.0,    # Assuming no grammar errors
            'readability': textstat.flesch_reading_ease(text),
            'topic_coherence': 0.0     # Only one sentence, so coherence is 0
        }

        # Mock the necessary external dependencies
        with patch('language_tool_python.LanguageTool.check', return_value=[]), \
             patch('spellchecker.SpellChecker.unknown', return_value=set()), \
             patch('spacy.load') as mock_spacy_load:

            # Create a mock nlp object
            mock_nlp = MagicMock()
            mock_doc = MagicMock()
            mock_doc.sents = [MagicMock()]
            mock_nlp.return_value = mock_doc
            mock_spacy_load.return_value = mock_nlp

            result = assess_lexical_quality(text)
            self.assertAlmostEqual(result['spelling_accuracy'], expected_output['spelling_accuracy'])
            self.assertAlmostEqual(result['grammar_quality'], expected_output['grammar_quality'])
            self.assertAlmostEqual(result['readability'], expected_output['readability'])
            self.assertAlmostEqual(result['topic_coherence'], expected_output['topic_coherence'])

    def test_process_dataset_in_batches(self):
        dataset = [
            "The quick brown fox jumps over the lazy dog.",
            "In 2020, the global pandemic changed the way we live and work."
        ]

        expected_output = [
            {
                'spelling_accuracy': 1.0,
                'grammar_quality': 1.0,
                'readability': textstat.flesch_reading_ease(dataset[0]),
                'topic_coherence': 0.0
            },
            {
                'spelling_accuracy': 1.0,
                'grammar_quality': 1.0,
                'readability': textstat.flesch_reading_ease(dataset[1]),
                'topic_coherence': 0.0
            }
        ]

        # Mock the necessary external dependencies
        with patch('language_tool_python.LanguageTool.check', return_value=[]), \
             patch('spellchecker.SpellChecker.unknown', return_value=set()), \
             patch('spacy.load') as mock_spacy_load:

            # Create a mock nlp object
            mock_nlp = MagicMock()
            mock_doc = MagicMock()
            mock_doc.sents = [MagicMock()]
            mock_nlp.return_value = mock_doc
            mock_spacy_load.return_value = mock_nlp

            result = process_dataset_in_batches(dataset, batch_size=1)
            for res, exp in zip(result, expected_output):
                self.assertAlmostEqual(res['spelling_accuracy'], exp['spelling_accuracy'])
                self.assertAlmostEqual(res['grammar_quality'], exp['grammar_quality'])
                self.assertAlmostEqual(res['readability'], exp['readability'])
                self.assertAlmostEqual(res['topic_coherence'], exp['topic_coherence'])

if __name__ == '__main__':
    unittest.main()
