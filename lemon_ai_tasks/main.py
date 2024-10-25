from clean_lab_tutorial import (
    raw_train_texts,
    train_labels,
    raw_test_texts,
    test_labels,
    encoder,
    CleanLearning,
    SentenceTransformer,
    LogisticRegression,
    accuracy_score,
)
from refine_confidence_scores import get_lexical_quality_scores, filter_or_weigh_texts, adjust_confidence_with_lexical_quality

def extract_vector_embeddings(train_texts, test_texts):
    transformer = SentenceTransformer('all-MiniLM-L6-v2')
    return transformer.encode(train_texts), transformer.encode(test_texts)

if __name__ == '__main__':
    # Get lexical quality scores
    lexical_quality_scores = get_lexical_quality_scores(raw_train_texts)
    print("Lexical quality scores are calculated.")

    # Filter and weigh texts based on lexical quality
    filtered_texts_with_weights = filter_or_weigh_texts(raw_train_texts, lexical_quality_scores)
    print("Texts are filtered and weighed based on lexical quality.")
    # Separate texts and weights
    filtered_texts, weights = zip(*filtered_texts_with_weights)
    print("Texts and weights are separated.")
    # Extract vector embeddings for filtered texts
    train_texts, test_texts = extract_vector_embeddings(filtered_texts, raw_test_texts)
    print("Vector embeddings are extracted for filtered texts.")
    # Define a classification model and use cleanlab to find potential label errors
    model = LogisticRegression(max_iter=1)
    print("A classification model is defined.")

    cv_n_folds = 5  # for efficiency; values like 5 or 10 will generally work better

    cl = CleanLearning(model, cv_n_folds=cv_n_folds)
    print("Cleanlab is initialized.")
    # Find label issues
    label_issues = cl.find_label_issues(X=train_texts, labels=train_labels)
    print("Label issues are found.")
    # Adjust confidence scores using lexical quality
    original_confidences = label_issues["label_quality"].to_numpy()
    adjusted_confidences = adjust_confidence_with_lexical_quality(original_confidences, lexical_quality_scores)
    print("Confidence scores are adjusted using lexical quality.")
    # Update label issues with adjusted confidences
    label_issues["adjusted_label_quality"] = adjusted_confidences
    print("Label issues are updated with adjusted confidences.")
    identified_issues = label_issues[label_issues["is_label_issue"] == True]
    lowest_quality_labels = label_issues["adjusted_label_quality"].argsort()[:10].to_numpy()
    print("The top 10 most likely label errors are identified.")

    print(
        f"cleanlab found {len(identified_issues)} potential label errors in the dataset.\n"
        f"Here are indices of the top 10 most likely errors: \n {lowest_quality_labels}"
    )

    # Train a more robust model from noisy labels
    model.fit(X=train_texts, y=train_labels)
    print("A more robust model is trained from noisy labels.")

    preds = model.predict(test_texts)
    acc_og = accuracy_score(test_labels, preds)
    print(f"\n Test accuracy of original model: {acc_og}")

    cl.fit(X=train_texts, labels=train_labels, label_issues=cl.get_label_issues())

    pred_labels = cl.predict(test_texts)
    acc_cl = accuracy_score(test_labels, pred_labels)
    print(f"Test accuracy of cleanlab's model: {acc_cl}")
