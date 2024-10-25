import os

# Set TOKENIZERS_PARALLELISM to false to avoid warnings
os.environ["TOKENIZERS_PARALLELISM"] = "false"

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
from refine_confidence_scores import get_lexical_quality_scores, adjust_confidence_with_lexical_quality

if __name__ == '__main__':
    # Load a different SentenceTransformer model
    print("Loading SentenceTransformer model.")
    transformer = SentenceTransformer('all-MiniLM-L6-v2')

    train_texts = transformer.encode(raw_train_texts)
    test_texts = transformer.encode(raw_test_texts)

    print("Extracted vector embeddings for train and test texts.")

    # Get lexical quality scores
    lexical_quality_scores = get_lexical_quality_scores(raw_train_texts)
    print("Calculated lexical quality scores for train texts.")

    # Define a classification model and use cleanlab to find potential label errors
    model = LogisticRegression(max_iter=1)
    print("Defined a LogisticRegression model.")
    cv_n_folds = 5  # for efficiency; values like 5 or 10 will generally work better

    cl = CleanLearning(model, cv_n_folds=cv_n_folds)
    print("Initialized CleanLearning with the model and cross-validation folds.")
    # Find label issues
    label_issues = cl.find_label_issues(X=train_texts, labels=train_labels)
    print("Found label issues in the training data.")

    # Train a more robust model from noisy labels
    model.fit(X=train_texts, y=train_labels)
    print("Trained a more robust model from noisy labels.")

    preds = model.predict(test_texts)
    acc_og = accuracy_score(test_labels, preds)
    print(f"\nTest accuracy of original model: {acc_og}")

    # Fit cleanlab model before adjusting confidences
    cl.fit(X=train_texts, labels=train_labels, label_issues=cl.get_label_issues())
    pred_labels_before_adjustment = cl.predict(test_texts)
    acc_cl_before = accuracy_score(test_labels, pred_labels_before_adjustment)
    print(f"Test accuracy of cleanlab's model before adjusting confidences: {acc_cl_before}")

    # Adjust confidence scores using lexical quality
    original_confidences = label_issues["label_quality"].to_numpy()
    adjusted_confidences = adjust_confidence_with_lexical_quality(original_confidences, lexical_quality_scores)

    # Update label issues with adjusted confidences
    label_issues["adjusted_label_quality"] = adjusted_confidences

    # Fit cleanlab model after adjusting confidences
    cl.fit(X=train_texts, labels=train_labels, label_issues=label_issues)
    pred_labels_after_adjustment = cl.predict(test_texts)
    acc_cl_after = accuracy_score(test_labels, pred_labels_after_adjustment)
    print(f"Test accuracy of cleanlab's model after adjusting confidences: {acc_cl_after}")
