import re
import string
import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split, cross_val_predict
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression
from sentence_transformers import SentenceTransformer

from cleanlab.classification import CleanLearning

# load and format the text dataset
data = pd.read_csv("https://s.cleanlab.ai/banking-intent-classification.csv")
data.head()

raw_texts, raw_labels = data["text"].values, data["label"].values

raw_train_texts, raw_test_texts, raw_train_labels, raw_test_labels = train_test_split(raw_texts, raw_labels, test_size=0.1)

num_classes = len(set(raw_train_labels))
"""
print(f"This dataset has {num_classes} classes.")
print(f"Classes: {set(raw_train_labels)}")

i = 0
print(f"Example Label: {raw_train_labels[i]}")
print(f"Example Text: {raw_train_texts[i]}") """

# We will use sklearn’s LabelEncoder to encode our labels.
encoder = LabelEncoder()
encoder.fit(raw_train_labels)

train_labels = encoder.transform(raw_train_labels)
test_labels = encoder.transform(raw_test_labels)

if __name__ == '__main__':
    # Load a different SentenceTransformer model
    transformer = SentenceTransformer('all-MiniLM-L6-v2')

    train_texts = transformer.encode(raw_train_texts)
    test_texts = transformer.encode(raw_test_texts)

    # Define a classification model and use cleanlab to find potential label errors
    model = LogisticRegression(max_iter=1)

    cv_n_folds = 5  # for efficiency; values like 5 or 10 will generally work better

    cl = CleanLearning(model, cv_n_folds=cv_n_folds)

    label_issues = cl.find_label_issues(X=train_texts, labels=train_labels)

    identified_issues = label_issues[label_issues["is_label_issue"] == True]
    lowest_quality_labels = label_issues["label_quality"].argsort()[:10].to_numpy()

    print(
        f"cleanlab found {len(identified_issues)} potential label errors in the dataset.\n"
        f"Here are indices of the top 10 most likely errors: \n {lowest_quality_labels}"
    )

    def print_as_df(index):
        return pd.DataFrame(
            {
                "text": raw_train_texts,
                "given_label": raw_train_labels,
                "predicted_label": encoder.inverse_transform(label_issues["predicted_label"]),
            },
        ).iloc[index]

    print("Top 5 most likely label errors:")
    print(print_as_df(lowest_quality_labels[:5]))

    # train a more robust model from noisy labels


    model.fit(X=train_texts, y=train_labels)

    preds = model.predict(test_texts)
    acc_og = accuracy_score(test_labels, preds)
    print(f"\n Test accuracy of original model: {acc_og}")

    cl.fit(X=train_texts, labels=train_labels, label_issues=cl.get_label_issues())

    pred_labels = cl.predict(test_texts)
    acc_cl = accuracy_score(test_labels, pred_labels)
    print(f"Test accuracy of cleanlab's model: {acc_cl}")



