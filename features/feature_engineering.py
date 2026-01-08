import re
import math
import numpy as np
from collections import Counter

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import RobustScaler
from scipy.sparse import hstack


# ======================================================
# 1. SAFE UTILITIES
# ======================================================

def safe_text(x):
    """Handle missing / null text fields safely."""
    if x is None:
        return ""
    if not isinstance(x, str):
        return str(x)
    return x


def safe_log(x):
    return math.log1p(max(x, 0))


# ======================================================
# 2. NOISE-AWARE TEXT PREPROCESSING
# ======================================================

BOILERPLATE_PATTERNS = [
    r"time limit.*",
    r"memory limit.*",
    r"input file.*",
    r"output file.*",
    r"constraints.*",
    r"all test cases.*",
    r"note that.*",
]

def preprocess_text(text: str) -> str:
    text = safe_text(text).lower()

    # Remove platform boilerplate
    for pat in BOILERPLATE_PATTERNS:
        text = re.sub(pat, " ", text)

    # Normalize math, operators, numbers
    text = re.sub(r"[≤≥<>]", " COMP ", text)
    text = re.sub(r"[+\-*/%=]", " OP ", text)
    text = re.sub(r"\d+", " NUM ", text)

    # Remove punctuation clutter
    text = re.sub(r"[\[\]\(\)\{\},;:]", " ", text)

    # Normalize whitespace
    text = re.sub(r"\s+", " ", text)

    return text.strip()


def combine_fields(row: dict) -> str:
    combined = " ".join([
        safe_text(row.get("title")),
        safe_text(row.get("description")),
        safe_text(row.get("input_description")),
        safe_text(row.get("output_description"))
    ])
    return preprocess_text(combined)


# ======================================================
# 3. ADVANCED HANDCRAFTED DIFFICULTY SIGNALS
# ======================================================

ALGO_KEYWORDS = {
    "dp": ["dp", "dynamic programming"],
    "graph": ["graph", "tree", "bfs", "dfs", "shortest path"],
    "greedy": ["greedy"],
    "search": ["binary search", "search"],
    "math": ["modulo", "gcd", "lcm", "prime"],
    "ds": ["segment tree", "fenwick", "heap", "stack", "queue"],
}

FAILURE_SIGNALS = [
    "be careful",
    "note that",
    "important",
    "edge case",
    "watch out",
    "pay attention"
]


def lexical_entropy(words):
    counts = Counter(words)
    total = len(words)
    if total == 0:
        return 0.0
    probs = [c / total for c in counts.values()]
    return -sum(p * math.log(p + 1e-9) for p in probs)


def extract_numeric_features(text: str):
    words = text.split()
    word_count = len(words)
    char_count = len(text)

    # 1. Text length
    log_words = safe_log(word_count)
    log_chars = safe_log(char_count)

    # 2. Algorithmic keywords
    algo_hits = sum(
        any(k in text for k in keys)
        for keys in ALGO_KEYWORDS.values()
    )

    # 3. Distinct algorithm families
    algo_diversity = sum(
        1 for keys in ALGO_KEYWORDS.values()
        if any(k in text for k in keys)
    )

    # 4. Lexical entropy
    entropy = lexical_entropy(words)

    # 5. Cognitive load
    cognitive_load = entropy * log_words

    # 6. Constraint pressure
    numbers = re.findall(r"\d+", text)
    max_number = safe_log(max(map(int, numbers))) if numbers else 0
    comparisons = text.count("COMP")
    constraint_pressure = max_number + comparisons

    # 7. Grammar complexity
    grammar_tokens = [
        "for each", "multiple", "sequence", "array",
        "matrix", "set of", "series"
    ]
    grammar_complexity = sum(tok in text for tok in grammar_tokens)

    # 8. Failure-awareness
    warning_signals = sum(sig in text for sig in FAILURE_SIGNALS)

    # 9. Numeric density
    numeric_density = len(numbers) / (word_count + 1)

    # 10. Symbol density
    symbol_density = text.count("OP") / (word_count + 1)

    # 11. Sentence complexity proxy
    sentence_count = max(1, text.count("."))
    avg_sentence_len = word_count / sentence_count

    # 12. Structural hint density
    structure_tokens = ["if", "else", "while", "for", "return"]
    structure_density = sum(t in text for t in structure_tokens)

    return [
        log_words,
        log_chars,
        algo_hits,
        algo_diversity,
        entropy,
        cognitive_load,
        constraint_pressure,
        grammar_complexity,
        warning_signals,
        numeric_density,
        symbol_density,
        avg_sentence_len,
        structure_density
    ]



# ======================================================
# 4. NOISY SAMPLE DETECTION (TEXT-ONLY)
# ======================================================

def detect_noisy_samples(texts):
    """
    Identify extremely noisy / malformed samples using
    text-only heuristics (no label leakage).
    """
    lengths = np.array([len(t.split()) for t in texts])

    # Extremely short or extremely long descriptions
    low, high = np.percentile(lengths, [2, 98])

    valid_mask = (lengths >= low) & (lengths <= high)
    return valid_mask


# ======================================================
# 5. FULL FEATURE PIPELINE
# ======================================================

def build_features_train(data):
    """
    Unified preprocessing + noise handling + feature extraction.
    Returns:
        X, word_tfidf, char_tfidf, scaler
    """

    # -------------------------------
    # Combine + preprocess text
    # -------------------------------
    texts = [combine_fields(row) for row in data]

    # -------------------------------
    # Remove noisy samples (TEXT-ONLY)
    # -------------------------------
    valid_mask = detect_noisy_samples(texts)
    texts = [t for t, keep in zip(texts, valid_mask) if keep]
    data = [row for row, keep in zip(data, valid_mask) if keep]

    # -------------------------------
    # WORD TF-IDF (SEMANTIC)
    # -------------------------------
    word_tfidf = TfidfVectorizer(
        max_features=5000,
        ngram_range=(1, 2),
        min_df=5,
        max_df=0.85,
        sublinear_tf=True,
        stop_words="english"
    )

    # -------------------------------
    # CHAR TF-IDF (SYNTAX / FORMAT)
    # -------------------------------
    char_tfidf = TfidfVectorizer(
        analyzer="char",
        ngram_range=(3, 5),
        max_features=3000,
        min_df=5,
        sublinear_tf=True
    )

    X_word = word_tfidf.fit_transform(texts)
    X_char = char_tfidf.fit_transform(texts)

    X_text = hstack([X_word, X_char])

    # -------------------------------
    # NUMERIC FEATURES (ROBUST)
    # -------------------------------
    X_num_raw = np.array(
        [extract_numeric_features(t) for t in texts],
        dtype=np.float64
    )

    scaler = RobustScaler()
    X_num = scaler.fit_transform(X_num_raw)

    # -------------------------------
    # FINAL FEATURE MATRIX
    # -------------------------------
    X = hstack([X_text, X_num]).tocsr()

    return X, word_tfidf, char_tfidf, scaler
