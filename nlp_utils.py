"""
nlp_utils.py
------------
NLP preprocessing pipeline for user input before sending to Gemini.

Uses spaCy + NLTK when available (local dev), falls back to lightweight
string-based processing in serverless environments (e.g. Vercel) where
these libraries aren't installed.
"""

import re
import logging
from typing import NamedTuple

logger = logging.getLogger(__name__)

# --- Optional heavy dependencies ---

try:
    import spacy
    _nlp = spacy.load("en_core_web_sm")
    SPACY_AVAILABLE = True
    logger.info("spaCy loaded successfully.")
except Exception:
    _nlp = None
    SPACY_AVAILABLE = False
    logger.info("spaCy not available — using fallback NLP.")

try:
    import nltk
    from nltk.sentiment import SentimentIntensityAnalyzer
    nltk.download("vader_lexicon", quiet=True)
    _sia = SentimentIntensityAnalyzer()
    NLTK_AVAILABLE = True
except Exception:
    _sia = None
    NLTK_AVAILABLE = False

# --- Data container ---

class PreprocessedInput(NamedTuple):
    original: str           # Raw user text
    cleaned: str            # Lowercased, punctuation-stripped text
    lemmatized: str         # Key lemmas joined as a string
    key_concepts: list[str] # Named entities + important nouns/verbs
    sentiment: str          # "positive" | "negative" | "neutral"
    is_question: bool       # Did the user ask a question?
    word_count: int         # Token count

# --- Main entry point ---

def preprocess(text: str) -> PreprocessedInput:
    """
    Full NLP preprocessing pipeline.
    Uses spaCy/NLTK when available, otherwise falls back to basic processing.
    """
    if not text or not text.strip():
        return PreprocessedInput(
            original="", cleaned="", lemmatized="",
            key_concepts=[], sentiment="neutral",
            is_question=False, word_count=0,
        )

    if SPACY_AVAILABLE:
        return _preprocess_spacy(text)
    else:
        return _preprocess_fallback(text)


# --- spaCy-powered pipeline ---

def _preprocess_spacy(text: str) -> PreprocessedInput:
    original = text.strip()
    cleaned = re.sub(r"[^\w\s\?\!\.]", " ", original.lower())
    cleaned = re.sub(r"\s+", " ", cleaned).strip()

    doc = _nlp(original)

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

    entities = [ent.text for ent in doc.ents]
    nouns_verbs = [
        token.text
        for token in doc
        if token.pos_ in {"NOUN", "PROPN", "VERB"}
        and not token.is_stop
        and len(token.text) > 2
    ]
    seen: set[str] = set()
    key_concepts: list[str] = []
    for item in entities + nouns_verbs:
        if item.lower() not in seen:
            seen.add(item.lower())
            key_concepts.append(item)
    key_concepts = key_concepts[:8]

    if NLTK_AVAILABLE:
        scores = _sia.polarity_scores(original)
        compound = scores["compound"]
        sentiment = "positive" if compound >= 0.05 else "negative" if compound <= -0.05 else "neutral"
    else:
        sentiment = "neutral"

    is_question = (
        original.strip().endswith("?")
        or any(
            token.text.lower() in {"what", "why", "how", "when", "where", "who", "which",
                                    "is", "are", "do", "does", "can", "could", "should", "would"}
            for token in doc[:3]
        )
    )
    word_count = sum(1 for token in doc if not token.is_space and not token.is_punct)

    return PreprocessedInput(
        original=original, cleaned=cleaned, lemmatized=lemmatized,
        key_concepts=key_concepts, sentiment=sentiment,
        is_question=is_question, word_count=word_count,
    )


# --- Lightweight fallback (no spaCy/NLTK) ---

_QUESTION_STARTERS = {"what", "why", "how", "when", "where", "who", "which",
                       "is", "are", "do", "does", "can", "could", "should", "would"}

_STOPWORDS = {"the", "a", "an", "and", "or", "but", "in", "on", "at", "to",
              "for", "of", "with", "by", "from", "it", "its", "this", "that",
              "i", "you", "he", "she", "we", "they", "my", "your", "our"}

def _preprocess_fallback(text: str) -> PreprocessedInput:
    original = text.strip()
    cleaned = re.sub(r"[^\w\s\?\!\.]", " ", original.lower())
    cleaned = re.sub(r"\s+", " ", cleaned).strip()

    words = cleaned.split()
    word_count = len(words)

    # Key concepts: words that are capitalised in original or long non-stopwords
    key_concepts = [
        w for w in original.split()
        if len(w) > 3 and w.lower() not in _STOPWORDS
    ][:8]

    lemmatized = " ".join(
        w for w in words if w not in _STOPWORDS and len(w) > 2
    )

    is_question = (
        original.strip().endswith("?")
        or (words and words[0].lower() in _QUESTION_STARTERS)
    )

    return PreprocessedInput(
        original=original, cleaned=cleaned, lemmatized=lemmatized,
        key_concepts=key_concepts, sentiment="neutral",
        is_question=is_question, word_count=word_count,
    )
