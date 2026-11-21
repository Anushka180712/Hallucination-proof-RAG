"""Detect and redact personal information (PII).

Why redact instead of block?
    A user who pastes "my email is jane@example.com, when do resupply missions
    arrive?" has a legitimate question. Refusing it is unhelpful; passing the
    email on to an LLM provider and into the logs is a privacy leak. Replacing
    it with ``[EMAIL]`` keeps the question answerable and keeps the data out of
    every downstream system.

Detection is pattern-based (no model, no network) and tuned for precision:
card numbers must pass the Luhn checksum, so random 16-digit numbers in a
document are not mistaken for cards.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

_PATTERNS: list[tuple[str, re.Pattern[str]]] = [
    ("EMAIL", re.compile(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b")),
    ("CARD", re.compile(r"\b(?:\d[ -]?){13,19}\b")),
    ("SSN", re.compile(r"\b\d{3}-\d{2}-\d{4}\b")),
    # International numbers must start with "+", which keeps dates and measurements
    # ("14 March 2041", "1 800 kWh") from looking like phone numbers.
    ("PHONE", re.compile(r"(?<!\w)\+\d{1,3}[ .-]?\(?\d{1,4}\)?(?:[ .-]?\d{2,4}){2,3}(?!\w)")),
    ("PHONE", re.compile(r"(?<!\w)(?:\(\d{3}\)[ .-]?|\d{3}[ .-])\d{3}[ .-]\d{4}(?!\w)")),
    ("IP", re.compile(r"\b(?:(?:25[0-5]|2[0-4]\d|1?\d?\d)\.){3}(?:25[0-5]|2[0-4]\d|1?\d?\d)\b")),
]


@dataclass
class PIIFinding:
    """One detected item: its kind (EMAIL, PHONE, ...) and character span."""
    kind: str
    start: int
    end: int


def _luhn_ok(number: str) -> bool:
    digits = [int(d) for d in number if d.isdigit()]
    if not 13 <= len(digits) <= 19:
        return False
    total = 0
    for i, d in enumerate(reversed(digits)):
        if i % 2 == 1:
            d = d * 2 - 9 if d * 2 > 9 else d * 2
        total += d
    return total % 10 == 0


def find_pii(text: str) -> list[PIIFinding]:
    """Find all PII spans. Earlier patterns win overlaps, so a card is never also read as a phone."""
    found: list[PIIFinding] = []
    taken: list[tuple[int, int]] = []
    for kind, pat in _PATTERNS:  # order matters: cards before phones, so a card isn't read as a phone
        for m in pat.finditer(text):
            if kind == "CARD" and not _luhn_ok(m.group()):
                continue
            if any(m.start() < e and m.end() > s for s, e in taken):
                continue
            found.append(PIIFinding(kind, m.start(), m.end()))
            taken.append((m.start(), m.end()))
    return sorted(found, key=lambda f: f.start)


def redact(text: str) -> tuple[str, list[str]]:
    """Return (text with PII replaced by ``[KIND]``, kinds found)."""
    findings = find_pii(text)
    out, last = [], 0
    for f in findings:
        out.append(text[last:f.start])
        out.append(f"[{f.kind}]")
        last = f.end
    out.append(text[last:])
    return "".join(out), [f.kind for f in findings]


_PLACEHOLDER = r"\[(?:EMAIL|CARD|SSN|PHONE|IP)\]"
_PII_CLAUSE = re.compile(
    r"(?:\b(?:and|also)\s+)?\b(?:my|our|his|her|their)\s+(?:[\w-]+\s+){0,3}?(?:is|are|was)\s*:?\s*"
    + _PLACEHOLDER + r"[\s,;.]*",
    re.IGNORECASE,
)


def strip_redacted(text: str) -> str:
    """Remove whole "my email is [EMAIL]" clauses, then any leftover placeholders.

    The placeholders themselves would otherwise be searched for ("email",
    "phone") and could make a perfectly answerable question look off-topic.
    """
    text = _PII_CLAUSE.sub(" ", text)
    text = re.sub(_PLACEHOLDER, " ", text)
    text = re.sub(r"\s+([,.;?!])", r"\1", text)
    return re.sub(r"^[\s,;.]+|\s{2,}", lambda m: "" if m.start() == 0 else " ", text).strip()
