"""Text normalization utilities for ASR evaluation.

Handles cleaning of reference transcripts and normalizing both
reference and hypothesis text for fair WER comparison.
"""

import re
from typing import Optional


def remove_emphasis_markers(text: str) -> str:
    """Remove **bold** emphasis markers from text.

    Args:
        text: Text with **emphasis** markers

    Returns:
        Text with markers removed
    """
    return re.sub(r'\*\*(.+?)\*\*', r'\1', text)


def remove_paralinguistic_tags(text: str) -> str:
    """Remove [paralinguistic] tags like [laugh], [cough], [deep-breath].

    Args:
        text: Text with [tags]

    Returns:
        Text with tags removed
    """
    return re.sub(r'\[[\w\-]+\]', '', text)


def normalize_whitespace(text: str) -> str:
    """Normalize whitespace - collapse multiple spaces, strip edges.

    Args:
        text: Text with potentially irregular whitespace

    Returns:
        Text with normalized whitespace
    """
    # Replace multiple spaces with single space
    text = re.sub(r'\s+', ' ', text)
    return text.strip()


def normalize_punctuation(text: str, remove_all: bool = False) -> str:
    """Normalize or remove punctuation.

    Args:
        text: Input text
        remove_all: If True, remove all punctuation. If False, just normalize.

    Returns:
        Text with normalized/removed punctuation
    """
    if remove_all:
        # Remove all punctuation except apostrophes in contractions
        text = re.sub(r"[^\w\s']", '', text)
        # Clean up any orphaned apostrophes
        text = re.sub(r"\s'|'\s", ' ', text)
    else:
        # Normalize common punctuation variations
        text = text.replace('…', '...')
        text = text.replace('–', '-')
        text = text.replace('—', '-')
        # Normalize quotes
        text = re.sub(r'["""]', '"', text)
        text = re.sub(r"[''']", "'", text)

    return text


def clean_reference_transcript(text: str) -> str:
    """Clean a reference transcript by removing annotation artifacts.

    Removes:
    - **emphasis** markers
    - [paralinguistic] tags
    - Normalizes whitespace

    Args:
        text: Raw reference transcript from dataset

    Returns:
        Cleaned transcript
    """
    text = remove_emphasis_markers(text)
    text = remove_paralinguistic_tags(text)
    text = normalize_whitespace(text)
    return text


def normalize_for_wer(text: str, lowercase: bool = True, remove_punctuation: bool = True) -> str:
    """Normalize text for WER calculation.

    Standard normalization for fair ASR comparison:
    - Lowercase (optional)
    - Remove punctuation (optional)
    - Normalize whitespace

    Args:
        text: Input text (reference or hypothesis)
        lowercase: Convert to lowercase
        remove_punctuation: Remove punctuation marks

    Returns:
        Normalized text ready for WER calculation
    """
    if lowercase:
        text = text.lower()

    if remove_punctuation:
        text = normalize_punctuation(text, remove_all=True)

    text = normalize_whitespace(text)
    return text


def prepare_for_evaluation(
    reference: str,
    hypothesis: str,
    lowercase: bool = True,
    remove_punctuation: bool = True,
) -> tuple[str, str]:
    """Prepare reference and hypothesis for evaluation.

    Args:
        reference: Raw reference transcript (may have annotations)
        hypothesis: Model prediction
        lowercase: Convert to lowercase for comparison
        remove_punctuation: Remove punctuation for comparison

    Returns:
        Tuple of (cleaned_reference, cleaned_hypothesis)
    """
    # Clean reference (remove annotations)
    ref_clean = clean_reference_transcript(reference)

    # Normalize both for WER
    ref_normalized = normalize_for_wer(ref_clean, lowercase, remove_punctuation)
    hyp_normalized = normalize_for_wer(hypothesis, lowercase, remove_punctuation)

    return ref_normalized, hyp_normalized
