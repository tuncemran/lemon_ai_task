import os
import torch
import random
import numpy as np
from torchtext.datasets import AG_NEWS
from torchtext.data.utils import get_tokenizer
from torchtext.vocab import build_vocab_from_iterator
from clean_lab_tutorial import (
    CleanLearning,
    SentenceTransformer,
    LogisticRegression,
    accuracy_score,
)
from refine_confidence_scores import get_lexical_quality_scores, adjust_confidence_with_lexical_quality
from torch.utils.data import DataLoader

# Set TOKENIZERS_PARALLELISM to false to avoid warnings
os.environ["TOKENIZERS_PARALLELISM"] = "false"

def introduce_text_noise(text, spelling_error_rate=0.1, grammar_error_rate=0.1):
    words = text.split()

    # Introduce spelling errors
    for i in range(len(words)):
        if random.random() < spelling_error_rate:
            # Simple spelling error: swap two letters
            if len(words[i]) > 1:
                char_list = list(words[i])
                idx = random.randint(0, len(char_list) - 2)
                char_list[idx], char_list[idx + 1] = char_list[idx + 1], char_list[idx]
                words[i] = ''.join(char_list)

    # Introduce grammar errors
    if random.random() < grammar_error_rate:
        # Simple grammar error: shuffle words
        random.shuffle(words)

    return ' '.join(words)

def load_ag_news_data(text_noise_level=0.1, fraction=0.25):
    # Load the AG News dataset
    train_iter = AG_NEWS(split='train')
    test_iter = AG_NEWS(split='test')

    # Convert iterators to lists
    train_data = list(train_iter)
    test_data = list(test_iter)

    # Shuffle and take a fraction of the data
    random.shuffle(train_data)
    random.shuffle(test_data)
    train_data = train_data[:int(len(train_data) * fraction)]
    test_data = test_data[:int(len(test_data) * fraction)]

    # Introduce text noise
    def introduce_noise(data, text_noise_level):
        noisy_data = []
        for label, text in data:
            if random.random() < text_noise_level:
                text = introduce_text_noise(text)
            noisy_data.append((label, text))
        return noisy_data

    noisy_train_data = introduce_noise(train_data, text_noise_level)
    noisy_test_data = introduce_noise(test_data, text_noise_level)

    # Tokenizer
    tokenizer = get_tokenizer('basic_english')

    # Build vocabulary
    def yield_tokens(data_iter):
        for _, text in data_iter:
            yield tokenizer(text)

    vocab = build_vocab_from_iterator(yield_tokens(noisy_train_data), specials=["<unk>"])
    vocab.set_default_index(vocab["<unk>"])

    # Process data
    def process_data(data):
        labels, texts = [], []
        for label, text in data:
            labels.append(label - 1)  # Adjust labels to be zero-indexed
            texts.append(vocab(tokenizer(text)))
        return labels, texts

    train_labels, train_texts = process_data(noisy_train_data)
    test_labels, test_texts = process_data(noisy_test_data)

    return train_labels, train_texts, test_labels, test_texts, vocab

# Convert token IDs back to strings
def tokens_to_texts(token_ids_list, vocab):
    return [' '.join([vocab.lookup_token(token_id) for token_id in token_ids]) for token_ids in token_ids_list]

def encode_texts_in_batches(texts, transformer, batch_size=32):
    dataloader = DataLoader(texts, batch_size=batch_size, shuffle=False)
    encoded_texts = []
    print(f"Encoding {len(texts)} texts in batches of {batch_size}...")
    for batch in dataloader:
        encoded_batch = transformer.encode(batch, convert_to_tensor=True, show_progress_bar=True)
        encoded_texts.extend(encoded_batch)
        print(f"Encoded {len(encoded_texts)} texts so far.")
    return encoded_texts

if __name__ == '__main__':
    # Load AG News data with a quarter of the dataset
    train_labels, train_texts, test_labels, test_texts, vocab = load_ag_news_data(fraction=0.01)
    print("Loaded a quarter of the AG News dataset.")

    # Convert token IDs back to strings
    original_train_texts = tokens_to_texts(train_texts, vocab)
    original_test_texts = tokens_to_texts(test_texts, vocab)

    # Load a different SentenceTransformer model
    print("Loading SentenceTransformer model.")
    transformer = SentenceTransformer('all-MiniLM-L6-v2')

    # Encode texts in batches
    train_texts = encode_texts_in_batches(original_train_texts, transformer)
    test_texts = encode_texts_in_batches(original_test_texts, transformer)

    print("Extracted vector embeddings for train and test texts.")

    # Convert train_texts and test_texts to NumPy arrays
    train_texts = np.array([t.cpu().numpy() for t in train_texts])
    test_texts = np.array([t.cpu().numpy() for t in test_texts])

    # Get lexical quality scores using the original texts
    lexical_quality_scores = get_lexical_quality_scores(original_train_texts)
    print("Calculated lexical quality scores for train texts.")

    # Define a classification model and use cleanlab to find potential label errors
    model = LogisticRegression(max_iter=100)
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

    # Predict using the test data
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
    print(f"Original confidences: {original_confidences}")
    adjusted_confidences = adjust_confidence_with_lexical_quality(original_confidences, lexical_quality_scores)
    print(f"Adjusted confidences: {adjusted_confidences}")

    # Update label issues with adjusted confidences
    label_issues["adjusted_label_quality"] = adjusted_confidences

    # Fit cleanlab model after adjusting confidences
    cl.fit(X=train_texts, labels=train_labels, label_issues=label_issues)
    pred_labels_after_adjustment = cl.predict(test_texts)
    acc_cl_after = accuracy_score(test_labels, pred_labels_after_adjustment)
    print(f"Test accuracy of cleanlab's model after adjusting confidences: {acc_cl_after}")
