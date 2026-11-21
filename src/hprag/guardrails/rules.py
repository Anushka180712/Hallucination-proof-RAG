"""Pattern rules for the guardrails, kept in one file so they are easy to review and extend.

Why rules instead of a classifier model?
    They run offline in microseconds, are deterministic (the same input always
    gets the same decision), and every block can be explained by pointing at the
    exact rule that fired. That transparency is the point of this project. The
    trade-off is coverage: a determined attacker can rephrase around a regex.
    The guardrails are therefore one layer, not the only one: even if an
    injection slips past them, the grounded prompt and claim verification still
    stop the model from presenting unsupported content as an answer.

Every pattern is phrase-level, not single words, to avoid false positives.
"fire" alone must not trigger anything (the demo corpus has a fire-research
program); "how do I make a bomb" must.
"""

from __future__ import annotations

import re

_I = re.IGNORECASE

# Attempts to override the system's instructions or change its role.
INJECTION_PATTERNS: list[tuple[str, re.Pattern[str]]] = [
    ("ignore-instructions", re.compile(
        r"\b(ignore|disregard|forget|override|bypass)\b.{0,30}\b(previous|prior|above|earlier|all|your|the|any)\b"
        r".{0,20}\b(instructions?|rules?|prompts?|guidelines|directions|context)\b", _I)),
    ("reveal-prompt", re.compile(
        # "your rules" is about the assistant; "the rules for spacewalks" is a normal question.
        r"\b(reveal|show|print|repeat|output|tell me|what (is|are))\b.{0,25}"
        r"\b(your (system prompt|instructions|hidden prompt|initial prompt|rules|guidelines)"
        r"|the (system prompt|hidden prompt|initial prompt))\b", _I)),
    ("role-override", re.compile(
        r"\b(you are now|from now on,? you|act as (an? )?(unfiltered|unrestricted|evil|jailbroken)|pretend (to be|you are)"
        r"|roleplay as)\b", _I)),
    ("jailbreak-keyword", re.compile(
        r"\b(jailbreak|DAN mode|do anything now|developer mode|god mode|no restrictions|without (any )?(rules|filters|restrictions))\b",
        _I)),
    ("answer-without-sources", re.compile(
        r"\b(use your own knowledge|answer anyway|even if (it'?s|it is) not in the (documents|sources)|make (something|it) up"
        r"|don'?t (use|cite) (the )?(sources|documents))\b", _I)),
    ("fake-system-tag", re.compile(r"(<\s*/?\s*(system|assistant|instructions?)\s*>|\[/?(SYSTEM|INST)\]|###\s*system)", _I)),
]

# Requests for seriously harmful content. Categories are kept explicit so the UI
# can say *what kind* of request was refused.
UNSAFE_PATTERNS: list[tuple[str, re.Pattern[str]]] = [
    ("weapons", re.compile(
        r"\b(how (do|can|to|would) (i|you|we|someone)|steps to|instructions for|recipe for|guide to)\b.{0,40}"
        r"\b(make|build|create|assemble|synthesi[sz]e|manufacture)\b.{0,30}\b(bomb|explosive|grenade|gun|firearm|weapon|nerve agent|poison)",
        _I)),
    ("cyberattack", re.compile(
        r"\b(how (do|can|to) (i|you|we)|help me|write)\b.{0,30}\b(hack into|break into|steal (passwords?|credentials)"
        r"|ddos|ransomware|keylogger|malware|phishing (email|page|site))", _I)),
    ("violence", re.compile(r"\b(how (do|can|to) (i|you)|best way to)\b.{0,30}\b(kill|murder|hurt|attack|poison)\b.{0,20}"
                            r"\b(someone|a person|people|him|her|my)\b", _I)),
]

# Self-harm is handled differently: not a lecture, but a supportive message.
SELF_HARM = re.compile(
    r"\b(kill myself|end my life|suicide|want to die|hurt myself|self[- ]harm|don'?t want to (live|be alive))\b", _I)

SELF_HARM_MESSAGE = (
    "It sounds like you might be going through something really hard. This tool only answers questions "
    "about its documents, but you don't have to deal with this alone. Please consider reaching out to "
    "someone you trust, or to a local crisis line (for example 988 in the US, or findahelpline.com "
    "for other countries)."
)

# Instructions planted inside *documents* (indirect prompt injection).
DOCUMENT_INJECTION_PATTERNS: list[tuple[str, re.Pattern[str]]] = [
    *[p for p in INJECTION_PATTERNS if p[0] in ("ignore-instructions", "role-override", "fake-system-tag")],
    ("addressed-to-ai", re.compile(
        r"\b(AI|assistant|chatbot|language model|LLM|model)s?\b.{0,40}\b(must|should|are required to|always|never)\b"
        r".{0,40}\b(answer|respond|reply|say|tell|recommend|output|include)\b", _I)),
    ("when-asked", re.compile(r"\bwhen (asked|questioned) about\b.{0,60}\b(say|answer|respond|reply|tell)\b", _I)),
]

# A deliberately small, mild list. Masking (not blocking) keeps answers usable when a
# source document quotes rude language, while keeping it out of the UI.
PROFANITY = re.compile(r"\b(fuck\w*|shit\w*|bitch\w*|bastard\w*|asshole\w*|cunt\w*|dickhead\w*|motherfuck\w*)\b", _I)


def first_match(text: str, patterns: list[tuple[str, re.Pattern[str]]]) -> tuple[str, str] | None:
    for name, pat in patterns:
        m = pat.search(text)
        if m:
            return name, m.group(0)
    return None
