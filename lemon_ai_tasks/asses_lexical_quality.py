"""
1. Integrate Lexical Quality Metrics:
    - Implement functionality to assess lexical quality for each text in the dataset using metrics like spelling accuracy, grammar quality, coherence, and readability.
    - Ensure this logic is efficient and scalable for larger datasets.

    - It uses the language_tool_python library to check grammar and spelling errors,
    - the SpellChecker library to check for misspelled words,
    - textstat to calculate the readability of the text.
    - spaCy for coherence.
    - Used batch processing to handle large datasets efficiently.
"""

import language_tool_python
from spellchecker import SpellChecker
import textstat
from concurrent.futures import ThreadPoolExecutor
import spacy

# Load spaCy model for coherence
nlp = spacy.load('en_core_web_sm')
tool = language_tool_python.LanguageTool('en-US')
spell = SpellChecker()


def assess_lexical_quality(text):
    # Ensure 'text' is a string
    if not isinstance(text, str):
        raise ValueError("Input must be a string")

    # Split the text into words
    words = text.split()  # No arguments needed for splitting by whitespace

    # Spelling accuracy
    misspelled = spell.unknown(words)
    spelling_accuracy = 1 - len(misspelled) / len(words)



    # Grammar quality
    matches = tool.check(text)
    grammar_quality = 1 - len(matches) / len(words)


    # Readability
    readability = textstat.flesch_reading_ease(text)


    # Topic coherence using spaCy
    doc = nlp(text)
    sentences = list(doc.sents)
    topic_coherence = 0.5
    if len(sentences) > 1:
        # Calculate average similarity between consecutive sentences
        similarities = [
            sentences[i].similarity(sentences[i + 1])
            for i in range(len(sentences) - 1)
        ]
        topic_coherence = sum(similarities) / len(similarities)

    return {
        'spelling_accuracy': spelling_accuracy,
        'grammar_quality': grammar_quality,
        'readability': readability,
        'topic_coherence': topic_coherence
    }

def process_dataset_in_batches(dataset, batch_size=10):
    """
    Processes a dataset in batches and assesses the lexical quality of each text.

    :param dataset: list, a list of texts to assess
    :param batch_size: int, the number of texts to process in each batch
    :return: list, a list of dictionaries containing the spelling accuracy, grammar quality, readability, and topic coherence for each text
    """
    results = []
    with ThreadPoolExecutor() as executor:
        # Process in batches
        for i in range(0, len(dataset), batch_size):
            batch = dataset[i:i + batch_size]
            batch_results = list(executor.map(assess_lexical_quality, batch))
            results.extend(batch_results)
    return results


if __name__ == "__main__":
    # Example usage with a more diverse dataset
    dataset = [
        "The quick brown fox jumps over the lazy dog. This sentence is often used to test typing skills.",
        "In 2020, the global pandemic changed the way we live and work, leading to a significant increase in remote work.",
        "Artificial intelligence is transforming industries by automating tasks and providing insights through data analysis."
    ]

    quality_results = process_dataset_in_batches(dataset)
    print(quality_results)
