# Installation Guide

- Use python 3.8.10 to avoid issues with versions of the libraries used in the project.

## How to install dependencies

```
pip install -r requirements.txt
python -m spacy download en_core_web_sm
```

## How to run the script

```
python asses_lexical_quality.py
```

## Design Decisions

- The script `asses_lexical_quality.py` is designed to assess the lexical quality of a given text. It calculates the spelling accuracy, grammar quality, readability, and topic coherence of the text. Topic coherence is calculated by averaging the similarity between consecutive sentences in the text therefore it will be 0 if the text contains only one sentence.
- The script `refine_confidence_scores.py` is designed to refine the confidence scores of a given text based on its lexical quality. It uses the lexical quality scores to weigh or filter the texts.
