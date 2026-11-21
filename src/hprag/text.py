"""Small, dependency-free text utilities used across retrieval and verification.

Keeping tokenisation in one place matters: if BM25, the reranker and the
verifier all tokenised differently, their scores would disagree for reasons
that have nothing to do with meaning.
"""

from __future__ import annotations

import re

# A compact English stopword list. Stopwords carry little meaning, so dropping
# them stops "the" and "of" from making unrelated sentences look similar.
STOPWORDS = frozenset(
    """a an and are as at be been being but by can could did do does doing for from had has have
    having he her here hers him his how i if in into is it its itself just me more most my no nor
    not of off on once only or other our ours out over own same she should so some such than that
    the their theirs them then there these they this those through to too under until up very was
    we were what when where which while who whom why will with would you your yours also about
    any each few both after before again against all am because between during further
    go goes went get gets got make makes made take takes put come comes thing things way""".split()
)

# Words that flip meaning. A claim and its evidence must agree on these, or a
# lexical verifier would happily "support" the exact opposite statement.
NEGATIONS = frozenset({"not", "no", "never", "none", "neither", "nor", "cannot", "without", "n't"})

_TOKEN_RE = re.compile(r"[a-z0-9]+(?:[.'][a-z0-9]+)*")
_NUM_RE = re.compile(r"\b\d+(?:[.,]\d+)*%?")
ABBREVIATIONS = ("Dr", "Mr", "Mrs", "Ms", "Prof", "St", "Jr", "Sr", "vs", "etc", "e.g", "i.e", "No", "Inc", "Ltd")
_ABBR_GUARD = "".join(rf"(?<!\b{re.escape(a)}\.)" for a in ABBREVIATIONS)
_SENT_RE = re.compile(_ABBR_GUARD + r"(?<=[.!?])\s+(?=[A-Z0-9\"'(\[])")
_CITE_RE = re.compile(r"\[(\d+(?:\s*,\s*\d+)*)\]")


def tokenize(text: str) -> list[str]:
    """Lowercase word tokens (keeps numbers like ``3.5`` and ``o'neil`` intact)."""
    text = text.lower().replace("\u2019", "'")
    text = re.sub(r"'s\b", "", text)  # possessive: "halcyon's" -> "halcyon"
    return _TOKEN_RE.findall(text.replace("n't", " n't"))


def stem(token: str) -> str:
    """A tiny suffix-stripping stemmer.

    Good enough to match "launched"/"launches"/"launch" without pulling in NLTK.
    It is deliberately conservative: over-stemming creates false matches, and
    false matches are exactly what turns into undetected hallucinations.
    """
    if len(token) <= 3 or token.isdigit():
        return token
    for suf in ("ization", "ations", "ation", "ingly", "edly", "ing", "ies", "ied", "es", "ed", "ly", "s"):
        if token.endswith(suf) and len(token) - len(suf) >= 3:
            base = token[: -len(suf)]
            return base + "y" if suf in ("ies", "ied") else base
    return token


# Words that shape a *question* ("how many", "how long") but are not about its topic.
# Leaving them in would make "How long ..." look for the word "long" in the corpus.
QUESTION_WORDS = frozenset(
    "many much long often far tell explain describe please know kind type exactly happen happens "
    "percentage percent amount number".split()
)


def query_terms(text: str) -> list[str]:
    """Content terms of a question, minus question-shaping words."""
    return [t for t in content_terms(text) if t not in QUESTION_WORDS]


def ends_with_abbreviation(text: str) -> bool:
    return any(text.endswith(f" {a}.") or text == f"{a}." for a in ABBREVIATIONS)


def content_terms(text: str) -> list[str]:
    """Stemmed tokens with stopwords removed — the 'meaning-bearing' words."""
    return [stem(t) for t in tokenize(text) if t not in STOPWORDS and t not in NEGATIONS]


def numbers(text: str) -> set[str]:
    """Numbers in a text, normalised (``1,200`` -> ``1200``)."""
    return {n.replace(",", "") for n in _NUM_RE.findall(text)}


def has_negation(text: str) -> bool:
    return any(t in NEGATIONS for t in tokenize(text))


def split_sentences(text: str) -> list[str]:
    """Rule-based sentence splitter. Handles the common cases without a model."""
    out: list[str] = []
    # Lines are split first so a markdown heading never fuses with the sentence below it.
    for line in text.splitlines():
        line = re.sub(r"\s+", " ", line).strip()
        if line:
            out.extend(s.strip() for s in _SENT_RE.split(line) if s.strip())
    return out


def extract_citations(text: str) -> tuple[str, list[int]]:
    """Strip ``[1]`` / ``[1, 2]`` markers from text and return (clean_text, numbers)."""
    nums: list[int] = []
    for m in _CITE_RE.finditer(text):
        nums.extend(int(n) for n in m.group(1).split(","))
    clean = _CITE_RE.sub("", text)
    clean = re.sub(r"\s+([.,;:!?])", r"\1", clean)
    return re.sub(r"\s{2,}", " ", clean).strip(), sorted(set(nums))
