"""
nlp_utils.py
------------
NLP preprocessing pipeline for user input before sending to Gemini.
Steps: lowercase → tokenize → remove stopwords → lemmatize → extract key concepts
"""

import re
import logging
from typing import NamedTuple

import spacy
import nltk
from nltk.sentiment import SentimentIntensityAnalyzer

logger = logging.getLogger(__name__)

# One-time model loading (done at import time so it's fast per-request)

try:
    nlp = spacy.load("en_core_web_sm")
except OSError:
    raise RuntimeError(
        "spaCy model 'en_core_web_sm' not found. "
        "Run: python -m spacy download en_core_web_sm"
    )

# Download VADER lexicon if missing (silent after first run)
nltk.download("vader_lexicon", quiet=True)
sia = SentimentIntensityAnalyzer()


# Data container for preprocessed results

class PreprocessedInput(NamedTuple):
    original: str           # Raw user text
    cleaned: str            # Lowercased, punctuation-stripped text
    lemmatized: str         # Key lemmas joined as a string
    key_concepts: list[str] # Named entities + important nouns/verbs
    sentiment: str          # "positive" | "negative" | "neutral"
    is_question: bool       # Did the user ask a question?
    word_count: int         # Token count (for length-aware prompting)


# Main preprocessing function

def preprocess(text: str) -> PreprocessedInput:
    """
    Full NLP preprocessing pipeline.

    Args:
        text: Raw user message string.

    Returns:
        PreprocessedInput with all derived fields.
    """
    if not text or not text.strip():
        return PreprocessedInput(
            original="",
            cleaned="",
            lemmatized="",
            key_concepts=[],
            sentiment="neutral",
            is_question=False,
            word_count=0,
        )

    # 1. Basic cleaning
    original = text.strip()
    cleaned = re.sub(r"[^\w\s\?\!\.]", " ", original.lower())
    cleaned = re.sub(r"\s+", " ", cleaned).strip()

    # 2. spaCy pipeline: tokenize, POS tag, NER, lemmatize
    doc = nlp(original)

    # 3. Extract lemmas (exclude stopwords, punctuation, whitespace, short tokens)
    meaningful_lemmas = [
        token.lemma_.lower()
        for token in doc
        if not token.is_stop
        and not token.is_punct
        and not token.is_space
        and len(token.text) > 2
        and token.pos_ in {"NOUN", "VERB", "ADJ", "ADV", "PROPN"}
    ]
    lemmatized = " ".join(meaningful_lemmas)

    # 4. Key concepts: named entities first, then top nouns/verbs
    entities = [ent.text for ent in doc.ents]
    nouns_verbs = [
        token.text
        for token in doc
        if token.pos_ in {"NOUN", "PROPN", "VERB"}
        and not token.is_stop
        and len(token.text) > 2
    ]
    # Deduplicate while preserving order
    seen: set[str] = set()
    key_concepts: list[str] = []
    for item in entities + nouns_verbs:
        if item.lower() not in seen:
            seen.add(item.lower())
            key_concepts.append(item)
    key_concepts = key_concepts[:8]  # cap at 8 concepts

    # 5. Sentiment via VADER
    scores = sia.polarity_scores(original)
    compound = scores["compound"]
    if compound >= 0.05:
        sentiment = "positive"
    elif compound <= -0.05:
        sentiment = "negative"
    else:
        sentiment = "neutral"

    # 6. Question detection
    is_question = (
        original.strip().endswith("?")
        or any(
            token.text.lower() in {"what", "why", "how", "when", "where", "who", "which", "is", "are", "do", "does", "can", "could", "should", "would"}
            for token in doc[:3]  # check first 3 tokens only
        )
    )

    # 7. Word count (real tokens, not whitespace splits)
    word_count = sum(1 for token in doc if not token.is_space and not token.is_punct)

    logger.debug(
        "Preprocessed: lemmas=%r | concepts=%r | sentiment=%s | question=%s",
        lemmatized,
        key_concepts,
        sentiment,
        is_question,
    )

    return PreprocessedInput(
        original=original,
        cleaned=cleaned,
        lemmatized=lemmatized,
        key_concepts=key_concepts,
        sentiment=sentiment,
        is_question=is_question,
        word_count=word_count,
    )
