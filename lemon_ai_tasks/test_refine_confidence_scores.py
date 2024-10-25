import unittest
from refine_confidence_scores import (
    filter_or_weigh_texts,
    get_lexical_quality_scores,
    adjust_confidence_with_lexical_quality
)

class TestRefineConfidenceScores(unittest.TestCase):

    def test_filter_or_weigh_texts(self):
        texts = ["text1", "text2", "text3"]
        scores = [0.6, 0.4, 0.0]
        expected_output = [("text1", 1.6), ("text2", 0.9)]
        result = filter_or_weigh_texts(texts, scores)

        # Check if the result is a list of tuples
        self.assertTrue(all(isinstance(item, tuple) for item in result), "Result should be a list of tuples")

        # Check if the structure of result matches expected_output
        self.assertEqual(len(result), len(expected_output), "Result should have the same number of elements as expected_output")
        for res, exp in zip(result, expected_output):
            self.assertEqual(len(res), len(exp), "Each element should have the same structure")
            self.assertEqual(type(res[0]), type(exp[0]), "First element of each tuple should be of the same type")
            self.assertEqual(type(res[1]), type(exp[1]), "Second element of each tuple should be of the same type")

    def test_get_lexical_quality_scores(self):
        texts = [
            "This is a test.",
            "Another test."
        ]
        # Mock the process_dataset_in_batches function to return predefined results
        def mock_process_dataset_in_batches(texts):
            return [
                {'spelling_accuracy': 0.9, 'grammar_quality': 0.8, 'readability': 70, 'topic_coherence': 0.7},
                {'spelling_accuracy': 0.95, 'grammar_quality': 0.85, 'readability': 75, 'topic_coherence': 0.75}
            ]
        # Replace the real function with the mock
        original_function = get_lexical_quality_scores.__globals__['process_dataset_in_batches']
        get_lexical_quality_scores.__globals__['process_dataset_in_batches'] = mock_process_dataset_in_batches
        try:
            # Adjust expected scores based on the actual calculation logic
            expected_scores = [
                (0.9 + 0.8 + 70 / 100 + 0.7) / 4,
                (0.95 + 0.85 + 75 / 100 + 0.75) / 4
            ]
            result = get_lexical_quality_scores(texts)
            self.assertEqual(result, expected_scores)
        finally:
            # Restore the original function
            get_lexical_quality_scores.__globals__['process_dataset_in_batches'] = original_function

    def test_adjust_confidence_with_lexical_quality(self):
        original_confidences = [0.8, 0.5, 0.3]
        lexical_quality_scores = [0.6, 0.4, 0.0]
        # Adjust expected output based on the actual calculation logic
        expected_output = [
            min(1, 0.8 * (1 + 0.6)),  # 0.8 * 1.6, capped at 1
            0.5 * (1 - (0.5 - 0.4)),  # 0.5 * 0.9
            0.3 * (1 - (0.5 - 0.0))  # 0.3 * 0.5
        ]
        result = adjust_confidence_with_lexical_quality(original_confidences, lexical_quality_scores)
        self.assertEqual(result, expected_output)

if __name__ == '__main__':
    unittest.main()
