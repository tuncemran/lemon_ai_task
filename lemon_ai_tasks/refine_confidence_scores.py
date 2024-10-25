from asses_lexical_quality import process_dataset_in_batches


def filter_or_weigh_texts(texts, lexical_quality_scores, threshold=0.5):
    """
    Filters or weighs texts based on their lexical quality scores.

    :param texts: list of str, the texts to be evaluated
    :param lexical_quality_scores: list of float, the lexical quality scores corresponding to each text
    :param threshold: float, the quality threshold for filtering
    :return: list of tuples, each containing a text and its weight
    """
    filtered_texts = []

    for text, quality_score in zip(texts, lexical_quality_scores):
        if quality_score >= threshold:
            # Assign a higher weight to high-quality texts
            weight = 1 + quality_score

        elif quality_score == 0:
            weight = 0
        else:
            # Assign a lower weight to low-quality texts
            weight = 1 - (threshold - quality_score)

        if weight > 0:
            filtered_texts.append((text, weight))

    return filtered_texts

def get_lexical_quality_scores(texts):
    """
    Gets lexical quality scores for a list of texts.

    :param texts: list of str, the texts to be evaluated
    :return: list of float, the lexical quality scores for each text
    """
    # Process the dataset to get quality metrics
    quality_metrics = process_dataset_in_batches(texts)

    # Calculate a composite lexical quality score for each text
    # Quality score is 0 if the text is empty or contains only one sentence or spelling accuracy is below 0.5 or grammar quality is below 0.5
    lexical_quality_scores = [
        0 if not metrics or len(metrics) <= 1 or metrics['spelling_accuracy'] < 0.5 or metrics['grammar_quality'] < 0.5 or metrics['topic_coherence'] < 0.5
        else (metrics['spelling_accuracy'] + metrics['grammar_quality'] + metrics['readability'] / 100 + metrics['topic_coherence']) / 4
        for metrics in quality_metrics
    ]

    return lexical_quality_scores

def adjust_confidence_with_lexical_quality(original_confidences, lexical_quality_scores, base_threshold=0.5):
    """
    Adjusts the confidence scores by incorporating the lexical quality scores.

    :param original_confidences: list of float, the original confidence scores from the model
    :param lexical_quality_scores: list of float, the lexical quality scores of the texts
    :param base_threshold: float, the base threshold for confidence
    :return: list of float, the adjusted confidence scores
    """
    adjusted_confidences = []

    for original_confidence, lexical_quality_score in zip(original_confidences, lexical_quality_scores):
        if lexical_quality_score < 0.5:
            # Increase the risk of mislabeling for low-quality texts
            adjusted_confidence = original_confidence * (1 - (0.5 - lexical_quality_score))
        else:
            # Lower the threshold for identifying label errors for high-quality texts
            adjusted_confidence = original_confidence * (1 + lexical_quality_score)

        # Ensure the adjusted confidence does not exceed 1 or drop below 0
        adjusted_confidence = max(0, min(1, adjusted_confidence))
        adjusted_confidences.append(adjusted_confidence)

    return adjusted_confidences

if __name__ == "__main__":
    # Example usage
    texts = [
    "The quick brown fox jumps over the lazy dog. This sentence is often used to test typing skills. It contains every letter of the alphabet.",
    "In 2020, the global pandemic changed the way we live and work. Remote work became the norm for many industries. People adapted to new technologies for communication.",
    "Artificial intelligence is transforming industries. It automates tasks and provides insights through data analysis. AI is expected to continue evolving rapidly.",
    "slnasl akldnad asd. ajndlcad. ddss dscsdc."
    ]
    lexical_quality_scores = get_lexical_quality_scores(texts)

    filtered_texts = filter_or_weigh_texts(texts, lexical_quality_scores)
    print(filtered_texts)
